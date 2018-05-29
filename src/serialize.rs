use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use std::fmt;
use std::marker::PhantomData;

macro_rules! impl_serde {
    ($name:ident) => {
        use $name;
        impl_serialize!($name);
        impl_deserialize!($name);
    };
}

macro_rules! impl_serialize {
    ($name:ident) => {
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_struct(stringify!($name), 0)?.end()
            }
        }
    };
}

macro_rules! impl_deserialize {
    ($name:ident) => {
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserializer.deserialize_struct(
                    stringify!($name),
                    &[],
                    DefaultVisitor::<$name>::new(),
                )
            }
        }
    };
}

impl_serde!(Node);
impl_serde!(Parameter);
impl_serde!(Tensor);

struct DefaultVisitor<T>(PhantomData<T>);

impl<T> DefaultVisitor<T> {
    pub fn new() -> Self {
        DefaultVisitor(PhantomData)
    }
}

impl<'de, T: Default> Visitor<'de> for DefaultVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("empty representation")
    }

    fn visit_seq<V>(self, seq: V) -> Result<T, V::Error>
    where
        V: SeqAccess<'de>,
    {
        let _ = seq;
        Ok(T::default())
    }

    fn visit_map<V>(self, mut map: V) -> Result<T, V::Error>
    where
        V: MapAccess<'de>,
    {
        assert!(
            map.next_key::<()>()?.is_none(),
            "object must not have any field"
        );
        Ok(T::default())
    }
}
