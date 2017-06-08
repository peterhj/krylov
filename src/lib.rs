//extern crate arraydiff;
extern crate densearray;
extern crate devicemem_cuda;
extern crate float;

extern crate rand;

//use arraydiff::prelude::*;
//use arraydiff::ops::*;
use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use float::ord::*;

use rand::*;
use rand::distributions::*;
use rand::distributions::normal::*;
use std::rc::{Rc};

pub struct LanczosEigensolver {
  max_iters:    usize,
  dim:      usize,
  q:        Vec<Vec<f32>>,
  z:        Vec<f32>,
  alpha:    Vec<f32>,
  beta:     Vec<f32>,
  //gamma:    Vec<f32>,
  //tri:      Array2d<f32>,
  evals:    Vec<f32>,
}

impl LanczosEigensolver {
  pub fn new(max_iters: usize, dim: usize) -> Self {
    let mut q = Vec::with_capacity(max_iters + 1);
    for _ in 0 .. max_iters + 1 {
      let mut q_i = Vec::with_capacity(dim);
      q_i.resize(dim, 0.0);
      q.push(q_i);
    }
    let mut z = Vec::with_capacity(dim);
    z.resize(dim, 0.0);
    let mut alpha = Vec::with_capacity(max_iters);
    alpha.resize(max_iters, 0.0);
    let mut beta = Vec::with_capacity(max_iters);
    beta.resize(max_iters, 0.0);
    /*let mut gamma = Vec::with_capacity(max_iters);
    gamma.resize(max_iters, 0.0);*/
    let mut evals = Vec::with_capacity(max_iters);
    evals.resize(max_iters, 0.0);
    LanczosEigensolver{
      max_iters:    max_iters,
      dim:      dim,
      q:        q,
      z:        z,
      alpha:    alpha,
      beta:     beta,
      //gamma:    gamma,
      //tri:      Array2d::zeros((max_iters, max_iters)),
      evals:    evals,
    }
  }

  pub fn solve<R, F>(&mut self, rng: &mut R, linear_fn: F) where R: Rng, F: Fn(&mut [f32], &mut [f32]) {
    // Lanczos iteration with full reorthogonalization.
    // See: <http://www.uta.edu/faculty/rcli/Teaching/math5371/DemmelMatlab/LANCZOS_README.html>.
    let q0_dist = Normal::new(0.0, 1.0);
    for j in 0 .. self.q[0].len() {
      self.q[0][j] = q0_dist.ind_sample(rng) as f32;
    }
    let q0_norm = self.q[0].flatten().l2_norm();
    self.q[0].flatten_mut().scale(1.0 / q0_norm);
    for iter_nr in 0 .. self.max_iters {
      linear_fn(&mut self.q[iter_nr], &mut self.z);
      self.alpha[iter_nr] = self.q[iter_nr].flatten().inner_prod(1.0, self.z.flatten());
      // Reorthogonalize twice just to be sure.
      for _ in 0 .. 2 {
        // Reorthogonalize using modified Gram-Schmidt.
        for i in 0 .. iter_nr + 1 {
          let proj_i = self.q[i].flatten().inner_prod(1.0, self.z.flatten());
          self.z.flatten_mut().add(-proj_i, self.q[i].flatten());
        }
      }
      self.beta[iter_nr] = self.z.flatten().l2_norm();
      self.q[iter_nr + 1].flatten_mut().copy(self.z.flatten());
      self.q[iter_nr + 1].flatten_mut().scale(1.0 / self.beta[iter_nr]);
    }

    match solve_symmetric_tridiagonal_eigenvalues(&self.alpha, &self.beta, &mut self.evals, 1.0e-6) {
      Err(_) => panic!("tridiagonal eigensolve failed"),
      Ok(neval) => {
        println!("DEBUG: LanczosEigensolver: number of eigenvalues: {}/{}", neval, self.dim);
      }
    }

    // Sort eigenvalues by decreasing magnitude.
    self.evals.sort_by(|&a, &b| {
      F32InfNan(-a.abs()).cmp(&F32InfNan(-b.abs()))
    });
  }

  pub fn largest_eigenvalue(&self) -> f32 {
    self.evals[0]
  }

  pub fn smallest_eigenvalue(&self) -> f32 {
    self.evals[self.max_iters - 1]
  }
}

pub struct ConjugateGradientConfig {
  pub max_iters:    usize,
  pub dim:          usize,
  pub dampener:     Option<f64>,
}

pub struct ConjugateGradientSolver32 {
  cfg:      ConjugateGradientConfig,
  b:        Array1d<f32>,
  b_norm:   f32,
  x:        Array1d<f32>,
  x_norm:   f32,
  r:        Array1d<f32>,
  r_norm:   f32,
  r_prev_norm:  f32,
  p_in:     Vec<f32>,
  p:        Array1d<f32>,
  w_out:    Vec<f32>,
  w:        Array1d<f32>,
}

impl ConjugateGradientSolver32 {
  pub fn solve<F>(&mut self, b: &[f32], linear_fn: F) where F: Fn(&mut Vec<f32>, &mut Vec<f32>) {
    assert_eq!(self.cfg.dim, b.len());
    self.b.as_view_mut().copy(b.reshape(self.cfg.dim));
    self.b_norm = self.b.as_view().l2_norm();
    self.x.as_view_mut().set_constant(0.0);
    self.x_norm = self.x.as_view().l2_norm();
    self.r.as_view_mut().copy(self.b.as_view());
    self.r_norm = self.r.as_view().l2_norm();
    println!("DEBUG: cg32: iter: {} |x|: {:.6} |r|: {:.6} |b|: {:.6}", 0, self.x_norm, self.r_norm, self.b_norm);
    for iter_nr in 0 .. self.cfg.max_iters {
      if iter_nr == 0 {
        self.p.as_view_mut().copy(self.r.as_view());
      } else {
        self.p.as_view_mut().scale(self.r_norm * self.r_norm / (self.r_prev_norm * self.r_prev_norm));
        self.p.as_view_mut().add(1.0, self.r.as_view());
      }
      self.p_in.reshape_mut(self.cfg.dim).copy(self.p.as_view());
      linear_fn(&mut self.p_in, &mut self.w_out);
      self.w.as_view_mut().copy(self.w_out.reshape(self.cfg.dim));
      if let Some(dampener) = self.cfg.dampener {
        self.w.as_view_mut().add(dampener as _, self.p.as_view());
      }
      let p_dot_w = self.p.as_view().inner_prod(1.0, self.w.as_view());
      if p_dot_w < 0.0 {
        println!("WARNING: cg32: p.w is negative, cg will probably fail");
      }
      let alpha = self.r_norm * self.r_norm / p_dot_w;
      self.x.as_view_mut().add(alpha, self.p.as_view());
      self.x_norm = self.x.as_view().l2_norm();
      self.r.as_view_mut().add(-alpha, self.w.as_view());
      self.r_prev_norm = self.r_norm;
      self.r_norm = self.r.as_view().l2_norm();
      println!("DEBUG: cg32: iter: {} |x|: {:.6} |r|: {:.6}", iter_nr + 1, self.x_norm, self.r_norm);
    }
  }
}

pub struct GPUConjugateGradientSolver32 {
  cfg:      ConjugateGradientConfig,
  b:        DeviceArray1d<f32>,
  b_norm:   f32,
  x:        DeviceArray1d<f32>,
  x_norm:   f32,
  r:        DeviceArray1d<f32>,
  r_norm:   f32,
  r_prev_norm:  f32,
  p_in:     DeviceMem<f32>,
  p:        DeviceArray1d<f32>,
  w_out:    DeviceMem<f32>,
  w:        DeviceArray1d<f32>,
}

// TODO
