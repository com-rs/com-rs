//! Nearest-point search in a finite set of complex points.
//!
//! [`Slicer`] finds the point of a set nearest to a complex value: by rounding the
//! coordinates when the points form a regular square (QAM) or linear (PAM) grid, in
//! `O(1)` per value, and by exhaustive search otherwise. It knows nothing of symbols or
//! bit labels; the constellations of `com-mapping` and the decision-directed loops of
//! `com-dsp` build on it.
//!
//! # Examples
//! ```
//! use com_math::{nearest::Slicer, C64};
//! let s = Slicer::new(&[C64::new(-1.0, 0.0), C64::new(1.0, 0.0)]);
//! assert_eq!(s.nearest(C64::new(0.3, 2.0)), 1);
//! assert!((s.distance2(C64::new(0.5, 0.0)) - 0.25).abs() < 1e-12);
//! ```

use crate::C64;

/// Nearest-point search in a set of points: by rounding the coordinates on a regular square (QAM) or
/// linear (PAM) grid, `O(1)` per sample, otherwise by exhaustive search.
#[derive(Clone, Debug, PartialEq)]
pub struct Slicer {
    points: Vec<C64>,
    /// `(side, step, first real level, first imaginary level, point of every cell)`.
    grid: Option<(usize, f64, f64, f64, Vec<usize>)>,
}

impl Slicer {
    /// Detect a regular grid among the points.
    pub fn new(points: &[C64]) -> Slicer {
        Slicer { points: points.to_vec(), grid: Slicer::detect(points) }
    }

    /// A slicer that always searches exhaustively (the reference of the grid search).
    pub fn exhaustive(points: &[C64]) -> Slicer {
        Slicer { points: points.to_vec(), grid: None }
    }

    /// Whether the points form a regular grid (the `O(1)` search is used).
    pub fn is_grid(&self) -> bool {
        self.grid.is_some()
    }

    /// The points.
    pub fn points(&self) -> &[C64] {
        &self.points
    }

    fn levels(v: impl Iterator<Item = f64>) -> Vec<f64> {
        let mut l: Vec<f64> = v.collect();
        l.sort_by(f64::total_cmp);
        let span = (l[l.len() - 1] - l[0]).abs().max(1e-300);
        l.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * span);
        l
    }

    fn regular(l: &[f64]) -> Option<f64> {
        if l.len() < 2 {
            return None;
        }
        let step = (l[l.len() - 1] - l[0]) / (l.len() - 1) as f64;
        l.iter().enumerate().all(|(i, v)| (v - (l[0] + step * i as f64)).abs() <= 1e-9 * step.abs()).then_some(step)
    }

    fn detect(points: &[C64]) -> Option<(usize, f64, f64, f64, Vec<usize>)> {
        let m = points.len();
        if m < 2 {
            return None;
        }
        let re = Slicer::levels(points.iter().map(|p| p.re));
        let step = Slicer::regular(&re)?;
        let tol = 1e-6 * step;
        let col = |x: f64| ((x - re[0]) / step).round() as usize;
        if points.iter().all(|p| p.im.abs() <= tol) {
            // a real grid (PAM): one row
            if re.len() != m {
                return None;
            }
            let mut lookup = vec![0; m];
            for (i, p) in points.iter().enumerate() {
                lookup[col(p.re)] = i;
            }
            return Some((m, step, re[0], 0.0, lookup));
        }
        let im = Slicer::levels(points.iter().map(|p| p.im));
        let step_im = Slicer::regular(&im)?;
        let side = re.len();
        if im.len() != side || side * side != m || (step_im - step).abs() > tol {
            return None;
        }
        let mut lookup = vec![usize::MAX; m];
        for (i, p) in points.iter().enumerate() {
            let (c, r) = (col(p.re), ((p.im - im[0]) / step).round() as usize);
            if c >= side || r >= side || lookup[r * side + c] != usize::MAX {
                return None;
            }
            lookup[r * side + c] = i;
        }
        Some((side, step, re[0], im[0], lookup))
    }

    /// Index of the nearest point.
    #[inline]
    pub fn nearest(&self, z: C64) -> usize {
        if let Some((side, step, re0, im0, lookup)) = &self.grid {
            let cell = |x: f64, x0: f64| ((x - x0) / step).round().clamp(0.0, (*side - 1) as f64) as usize;
            let c = cell(z.re, *re0);
            if lookup.len() == *side { lookup[c] } else { lookup[cell(z.im, *im0) * side + c] }
        } else {
            let mut best = (f64::INFINITY, 0usize);
            for (i, p) in self.points.iter().enumerate() {
                let d = (z - p).norm_sqr();
                if d < best.0 {
                    best = (d, i);
                }
            }
            best.1
        }
    }

    /// Squared distance to the nearest point.
    #[inline]
    pub fn distance2(&self, z: C64) -> f64 {
        (z - self.points[self.nearest(z)]).norm_sqr()
    }
}
