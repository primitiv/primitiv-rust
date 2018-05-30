#[allow(unused_imports)]
#[macro_use]
extern crate primitiv;
#[allow(unused_imports)]
#[macro_use]
extern crate serde_derive;
#[allow(unused_imports)]
extern crate serde_json;

#[cfg(feature = "serialize")]
mod tests {
    use serde_json;

    use primitiv::Model;
    use primitiv::Node;
    use primitiv::Parameter;
    use primitiv::Tensor;

    #[derive(Model, Serialize, Deserialize)]
    struct Model1 {
        pw1: Parameter,
        pw2: Parameter,
    }

    impl Model1 {
        pub fn new() -> Self {
            Model1 {
                pw1: Parameter::new(),
                pw2: Parameter::new(),
            }
        }
    }

    #[derive(Model, Serialize, Deserialize)]
    struct Model2 {
        #[primitiv(submodel)]
        model1: Model1,
        pw: Parameter,
    }

    impl Model2 {
        pub fn new() -> Self {
            Model2 {
                model1: Model1::new(),
                pw: Parameter::new(),
            }
        }
    }

    #[test]
    fn serde_test() {
        {
            let node = Node::new();
            let serialized = serde_json::to_string(&node).unwrap();
            assert_eq!("{}", serialized);
            let deserialized: Tensor = serde_json::from_str(&serialized).unwrap();
            assert!(!deserialized.valid());
        }
        {
            let param = Parameter::new();
            let serialized = serde_json::to_string(&param).unwrap();
            assert_eq!("{}", serialized);
            let deserialized: Parameter = serde_json::from_str(&serialized).unwrap();
            assert!(!deserialized.valid());
        }
        {
            let tensor = Tensor::new();
            let serialized = serde_json::to_string(&tensor).unwrap();
            assert_eq!("{}", serialized);
            let deserialized: Tensor = serde_json::from_str(&serialized).unwrap();
            assert!(!deserialized.valid());
        }
        {
            let mut model = Model1::new();
            assert!(model.get_parameter("pw1").is_none());
            assert!(model.get_parameter("pw2").is_none());
            model.register_parameters();
            assert!(model.get_parameter("pw1").is_some());
            assert!(model.get_parameter("pw2").is_some());
            let serialized = serde_json::to_string(&model).unwrap();
            assert_eq!("{\"pw1\":{},\"pw2\":{}}", serialized);
            let mut deserialized: Model1 = serde_json::from_str(&serialized).unwrap();
            assert!(deserialized.get_parameter("pw1").is_none());
            assert!(deserialized.get_parameter("pw2").is_none());
            deserialized.register_parameters();
            assert!(deserialized.get_parameter("pw1").is_some());
            assert!(deserialized.get_parameter("pw2").is_some());
            let serialized = serde_json::to_string(&deserialized).unwrap();
            assert_eq!("{\"pw1\":{},\"pw2\":{}}", serialized);
        }
        {
            let mut model = Model2::new();
            assert!(model.get_submodel("model1").is_none());
            assert!(model.find_parameter(&["model1", "pw1"]).is_none());
            assert!(model.find_parameter(&["model1", "pw2"]).is_none());
            assert!(model.get_parameter("pw").is_none());
            model.register_parameters();
            assert!(model.get_submodel("model1").is_some());
            assert!(model.find_parameter(&["model1", "pw1"]).is_some());
            assert!(model.find_parameter(&["model1", "pw2"]).is_some());
            assert!(model.get_parameter("pw").is_some());
            let serialized = serde_json::to_string(&model).unwrap();
            assert_eq!("{\"model1\":{\"pw1\":{},\"pw2\":{}},\"pw\":{}}", serialized);
            let mut deserialized: Model2 = serde_json::from_str(&serialized).unwrap();
            assert!(deserialized.get_submodel("model1").is_none());
            assert!(deserialized.find_parameter(&["model1", "pw1"]).is_none());
            assert!(deserialized.find_parameter(&["model1", "pw2"]).is_none());
            assert!(deserialized.get_parameter("pw").is_none());
            deserialized.register_parameters();
            assert!(deserialized.get_submodel("model1").is_some());
            assert!(deserialized.find_parameter(&["model1", "pw1"]).is_some());
            assert!(deserialized.find_parameter(&["model1", "pw2"]).is_some());
            assert!(deserialized.get_parameter("pw").is_some());
            let serialized = serde_json::to_string(&deserialized).unwrap();
            assert_eq!("{\"model1\":{\"pw1\":{},\"pw2\":{}},\"pw\":{}}", serialized);
        }
    }
}
