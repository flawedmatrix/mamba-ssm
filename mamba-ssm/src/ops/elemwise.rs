use candle::{bail, CpuStorage, CustomOp2, Layout, Shape};
use rayon::prelude::*;

pub enum BinaryOp {
    Add,
    Mul,
}

type C = CpuStorage;

impl CustomOp2 for BinaryOp {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "custom-add",
            Self::Mul => "custom-mul",
        }
    }

    /// The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
    /// offsets etc so the associated layout should be used to access it.
    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle::Result<(CpuStorage, Shape)> {
        if l1.shape() != l2.shape() {
            bail!("lhs and rhs mismatched shapes");
        }
        match (s1, s2) {
            (C::BF16(_v1), C::BF16(_v2)) => bail!("TODO: Implement BF16 binary op"),
            (C::F16(_v1), C::F16(_v2)) => bail!("TODO: Implement F16 binary op"),
            (C::F32(v1), C::F32(v2)) => Ok((C::F32(self.f(v1, l1, v2, l2)?), l1.shape().clone())),
            _ => bail!("Op not implemented"),
        }
    }
}

impl BinaryOp {
    fn f(
        &self,
        lhs: &[f32],
        lhs_l: &Layout,
        rhs: &[f32],
        rhs_l: &Layout,
    ) -> candle::Result<Vec<f32>> {
        match (lhs_l.contiguous_offsets(), rhs_l.contiguous_offsets()) {
            (Some((o_l1, o_l2)), Some((o_r1, o_r2))) => {
                let n = o_l2 - o_l1;
                let mut v: Vec<f32> = Vec::with_capacity(o_l2 - o_l1);
                unsafe {
                    v.set_len(n);
                }
                match self {
                    BinaryOp::Add => {
                        binary_add_f32(&lhs[o_l1..o_l2], &rhs[o_r1..o_r2], v.as_mut_slice())
                    }
                    BinaryOp::Mul => {
                        binary_mul_f32(&lhs[o_l1..o_l2], &rhs[o_r1..o_r2], v.as_mut_slice())
                    }
                };
                Ok(v)
            }
            _ => bail!("lhs and rhs layouts must be C contiguous"),
        }
    }
}

fn binary_add_f32(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
    let chunk_size = dst.len() / 8;
    let operand_chunks = lhs.par_chunks(chunk_size).zip(rhs.par_chunks(chunk_size));
    dst.par_chunks_mut(chunk_size).zip(operand_chunks).for_each(
        |(dst_chunk, (lhs_chunk, rhs_chunk))| {
            let operands = lhs_chunk.iter().zip(rhs_chunk.iter());
            dst_chunk
                .iter_mut()
                .zip(operands)
                .for_each(|(dst_val, (lhs, rhs))| {
                    *dst_val = lhs + rhs;
                });
        },
    );
}

fn binary_mul_f32(lhs: &[f32], rhs: &[f32], dst: &mut [f32]) {
    let chunk_size = dst.len() / 8;
    let operand_chunks = lhs.par_chunks(chunk_size).zip(rhs.par_chunks(chunk_size));
    dst.par_chunks_mut(chunk_size).zip(operand_chunks).for_each(
        |(dst_chunk, (lhs_chunk, rhs_chunk))| {
            let operands = lhs_chunk.iter().zip(rhs_chunk.iter());
            dst_chunk
                .iter_mut()
                .zip(operands)
                .for_each(|(dst_val, (lhs, rhs))| {
                    *dst_val = lhs * rhs;
                });
        },
    );
}

#[cfg(test)]
mod tests {
    use crate::ops::elemwise::{binary_add_f32, binary_mul_f32};

    #[test]
    fn test_binary_add_f32() {
        let lhs: Vec<f32> = (0i16..100).map(|i| i.into()).collect();
        let rhs: Vec<f32> = (0i16..100).rev().map(|i| i.into()).collect();
        let expected: Vec<f32> = vec![99.0; 100];
        let mut dst: Vec<f32> = vec![0.0; 100];

        binary_add_f32(lhs.as_slice(), rhs.as_slice(), dst.as_mut_slice());

        println!("Expected {expected:?}");
        for i in 0..100 {
            assert_eq!(dst[i], expected[i]);
        }
    }

    #[test]
    fn test_binary_mul_f32() {
        let lhs: Vec<f32> = (0i16..100).map(|i| i.into()).collect();
        let rhs: Vec<f32> = (0i16..100).rev().map(|i| i.into()).collect();
        let expected: Vec<f32> = (0i16..100)
            .zip((0i16..100).rev())
            .map(|(a, b)| (a * b).into())
            .collect();
        let mut dst: Vec<f32> = vec![0.0; 100];

        binary_mul_f32(lhs.as_slice(), rhs.as_slice(), dst.as_mut_slice());

        println!("Expected {expected:?}");
        for i in 0..100 {
            assert_eq!(dst[i], expected[i]);
        }
    }
}
