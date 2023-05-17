use spade::PointN;

use crate::{Scalar, ZERO_TRESHOLD};
use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

pub trait NavVec3:
    Sized + Clone + Copy + Add<Self, Output = Self> + Sub<Self, Output = Self>
{
    #[inline]
    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    fn new(x: Scalar, y: Scalar, z: Scalar) -> Self;

    fn x<'a>(&'a self) -> &'a Scalar;

    fn y<'a>(&'a self) -> &'a Scalar;

    fn z<'a>(&'a self) -> &'a Scalar;

    fn x_mut<'a>(&'a mut self) -> &'a mut Scalar;

    fn y_mut<'a>(&'a mut self) -> &'a mut Scalar;

    fn z_mut<'a>(&'a mut self) -> &'a mut Scalar;

    fn sqr_magnitude(self) -> Scalar;

    fn magnitude(self) -> Scalar;

    fn cross(self, rhs: Self) -> Self;

    fn dot(self, rhs: Self) -> Scalar;

    fn mul_elem(self, rhs: Self) -> Self;

    fn normalize(self) -> Self;

    fn coincides(self, other: Self) -> bool {
        (other - self).sqr_magnitude() < ZERO_TRESHOLD
    }

    fn project(self, from: Self, to: Self) -> Scalar {
        let diff = to - from;
        (self - from).dot(diff) / diff.sqr_magnitude()
    }

    fn unproject(from: Self, to: Self, t: Scalar) -> Self {
        let diff = to - from;
        from + Self::new(diff.x() * t, diff.y() * t, diff.z() * t)
    }

    fn min(self, other: Self) -> Self {
        Self::new(
            self.x().min(*other.x()),
            self.y().min(*other.y()),
            self.z().min(*other.z()),
        )
    }

    fn max(self, other: Self) -> Self {
        Self::new(
            self.x().max(*other.x()),
            self.y().max(*other.y()),
            self.z().max(*other.z()),
        )
    }

    fn distance_to_plane(self, origin: Self, normal: Self) -> Scalar {
        normal.dot(self - origin)
    }

    fn is_above_plane(self, origin: Self, normal: Self) -> bool {
        self.distance_to_plane(origin, normal) > -ZERO_TRESHOLD
    }

    fn project_on_plane(self, origin: Self, normal: Self) -> Self {
        let v = self - origin;
        let n = normal.normalize();
        let dot = v.dot(n);
        let d = Self::new(normal.x() * dot, normal.y() * dot, normal.z() * dot);
        self - d
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct SpadePoint<T>(pub(crate) T);

impl<T> PointN for SpadePoint<T>
where
    T: NavVec3 + PartialEq + Debug, // TODO
{
    type Scalar = Scalar;

    fn dimensions() -> usize {
        3
    }

    fn from_value(value: Self::Scalar) -> Self {
        SpadePoint(T::new(value, value, value))
    }

    fn nth(&self, index: usize) -> &Self::Scalar {
        match index {
            0 => self.0.x(),
            1 => self.0.y(),
            2 => self.0.z(),
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => self.0.x_mut(),
            1 => self.0.y_mut(),
            2 => self.0.z_mut(),
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "cgmath")]
pub mod opt_cgmath {
    use super::{NavVec3, Scalar};

    use cgmath::*;

    impl NavVec3 for Vector3<Scalar> {
        #[inline]
        fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
            Self { x, y, z }
        }

        #[inline]
        fn x(&self) -> &Scalar {
            &self.x
        }

        #[inline]
        fn y(&self) -> &Scalar {
            &self.y
        }

        #[inline]
        fn z(&self) -> &Scalar {
            &self.z
        }

        #[inline]
        fn x_mut(&mut self) -> &mut Scalar {
            &mut self.x
        }

        #[inline]
        fn y_mut(&mut self) -> &mut Scalar {
            &mut self.y
        }

        #[inline]
        fn z_mut(&mut self) -> &mut Scalar {
            &mut self.z
        }

        #[inline]
        fn sqr_magnitude(self) -> Scalar {
            self.magnitude2()
        }

        #[inline]
        fn magnitude(self) -> Scalar {
            cgmath::InnerSpace::magnitude(self)
        }

        #[inline]
        fn cross(self, rhs: Self) -> Self {
            self.cross(rhs)
        }

        #[inline]
        fn dot(self, rhs: Self) -> Scalar {
            cgmath::InnerSpace::dot(self, rhs)
        }

        #[inline]
        fn mul_elem(self, rhs: Self) -> Self {
            self.mul_element_wise(rhs)
        }

        #[inline]
        fn normalize(self) -> Self {
            cgmath::InnerSpace::normalize(self)
        }
    }
}

#[cfg(feature = "nalgebra")]
pub mod opt_nalgebra {
    use super::{NavVec3, Scalar};

    use nalgebra::*;

    impl NavVec3 for Vector3<Scalar> {
        #[inline]
        fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
            Self::new(x, y, z)
        }

        #[inline]
        fn x(&self) -> &Scalar {
            &self.x
        }

        #[inline]
        fn y(&self) -> &Scalar {
            &self.y
        }

        #[inline]
        fn z(&self) -> &Scalar {
            &self.z
        }

        #[inline]
        fn x_mut(&mut self) -> &mut Scalar {
            &mut self.x
        }

        #[inline]
        fn y_mut(&mut self) -> &mut Scalar {
            &mut self.y
        }

        #[inline]
        fn z_mut(&mut self) -> &mut Scalar {
            &mut self.z
        }

        #[inline]
        fn sqr_magnitude(self) -> Scalar {
            self.magnitude_squared()
        }

        #[inline]
        fn magnitude(self) -> Scalar {
            nalgebra::Vector3::magnitude(&self)
        }

        #[inline]
        fn cross(self, rhs: Self) -> Self {
            nalgebra::Vector3::cross(&self, &rhs)
        }

        #[inline]
        fn dot(self, rhs: Self) -> Scalar {
            nalgebra::Vector3::dot(&self, &rhs)
        }

        #[inline]
        fn mul_elem(self, rhs: Self) -> Self {
            self.component_mul(&rhs)
        }

        #[inline]
        fn normalize(self) -> Self {
            nalgebra::Vector3::normalize(&self)
        }
    }
}

#[cfg(feature = "glam")]
pub mod feature_glam {
    use super::{NavVec3, Scalar};

    impl NavVec3 for glam::Vec3 {
        #[inline]
        fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
            Self { x, y, z }
        }

        #[inline]
        fn x(&self) -> &Scalar {
            &self.x
        }

        #[inline]
        fn y(&self) -> &Scalar {
            &self.y
        }

        #[inline]
        fn z(&self) -> &Scalar {
            &self.z
        }

        #[inline]
        fn x_mut(&mut self) -> &mut Scalar {
            &mut self.x
        }

        #[inline]
        fn y_mut(&mut self) -> &mut Scalar {
            &mut self.y
        }

        #[inline]
        fn z_mut(&mut self) -> &mut Scalar {
            &mut self.z
        }

        #[inline]
        fn sqr_magnitude(self) -> Scalar {
            self.length_squared()
        }

        #[inline]
        fn magnitude(self) -> Scalar {
            self.length()
        }

        #[inline]
        fn cross(self, rhs: Self) -> Self {
            self.cross(rhs)
        }

        #[inline]
        fn dot(self, rhs: Self) -> Scalar {
            self.dot(rhs)
        }

        #[inline]
        fn mul_elem(self, rhs: Self) -> Self {
            self * rhs
        }

        #[inline]
        fn normalize(self) -> Self {
            self.normalize()
        }
    }

    impl NavVec3 for glam::Vec3A {
        #[inline]
        fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
            Self { x, y, z }
        }

        #[inline]
        fn x(&self) -> &Scalar {
            &self.x
        }

        #[inline]
        fn y(&self) -> &Scalar {
            &self.y
        }

        #[inline]
        fn z(&self) -> &Scalar {
            &self.z
        }

        #[inline]
        fn x_mut(&mut self) -> &mut Scalar {
            &mut self.x
        }

        #[inline]
        fn y_mut(&mut self) -> &mut Scalar {
            &mut self.y
        }

        #[inline]
        fn z_mut(&mut self) -> &mut Scalar {
            &mut self.z
        }

        #[inline]
        fn sqr_magnitude(self) -> Scalar {
            self.length_squared()
        }

        #[inline]
        fn magnitude(self) -> Scalar {
            self.length()
        }

        #[inline]
        fn cross(self, rhs: Self) -> Self {
            self.cross(rhs)
        }

        #[inline]
        fn dot(self, rhs: Self) -> Scalar {
            self.dot(rhs)
        }

        #[inline]
        fn mul_elem(self, rhs: Self) -> Self {
            self * rhs
        }

        #[inline]
        fn normalize(self) -> Self {
            self.normalize()
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NVec3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

impl From<[Scalar; 3]> for NVec3 {
    fn from(value: [Scalar; 3]) -> Self {
        NVec3 {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl Add for NVec3 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for NVec3 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<Scalar> for NVec3 {
    type Output = Self;

    #[inline]
    fn mul(self, other: Scalar) -> Self {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl Div<Scalar> for NVec3 {
    type Output = Self;

    #[inline]
    fn div(self, other: Scalar) -> Self {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl NavVec3 for NVec3 {
    fn new(x: Scalar, y: Scalar, z: Scalar) -> Self {
        Self { x, y, z }
    }

    fn x<'a>(&'a self) -> &'a Scalar {
        &self.x
    }

    fn y<'a>(&'a self) -> &'a Scalar {
        &self.y
    }

    fn z<'a>(&'a self) -> &'a Scalar {
        &self.z
    }

    fn x_mut<'a>(&'a mut self) -> &'a mut Scalar {
        &mut self.x
    }

    fn y_mut<'a>(&'a mut self) -> &'a mut Scalar {
        &mut self.y
    }

    fn z_mut<'a>(&'a mut self) -> &'a mut Scalar {
        &mut self.z
    }

    fn mul_elem(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }

    #[inline]
    fn sqr_magnitude(self) -> Scalar {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    fn magnitude(self) -> Scalar {
        self.sqr_magnitude().sqrt()
    }

    #[inline]
    fn coincides(self, other: Self) -> bool {
        (other - self).sqr_magnitude() < ZERO_TRESHOLD
    }

    #[inline]
    fn cross(self, other: Self) -> Self {
        Self {
            x: (self.y * other.z) - (self.z * other.y),
            y: (self.z * other.x) - (self.x * other.z),
            z: (self.x * other.y) - (self.y * other.x),
        }
    }

    #[inline]
    fn dot(self, other: Self) -> Scalar {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    fn normalize(self) -> Self {
        let len = self.magnitude();
        if len < ZERO_TRESHOLD {
            Self::new(0.0, 0.0, 0.0)
        } else {
            Self::new(self.x / len, self.y / len, self.z / len)
        }
    }
}
