#[macro_use]
extern crate primitiv;

use primitiv::Model;
use primitiv::Parameter;

#[derive(Model)]
struct Model1 {
    pw1: Parameter,
    pw2: Option<Parameter>,
    pw3: (Parameter, Parameter),
    pw4: [Parameter; 3],
    pw5: Vec<Parameter>,
    pw6: Option<Option<Parameter>>,
    pw7: Option<(Parameter, Parameter)>,
    pw8: Option<[Parameter; 3]>,
    pw9: Option<Vec<Parameter>>,
    pw10: (Option<Parameter>, Option<Parameter>),
    pw11: ((Parameter, Parameter), (Parameter, Parameter)),
    pw12: ([Parameter; 3], [Parameter; 3]),
    pw13: (Vec<Parameter>, Vec<Parameter>),
    pw14: [Option<Parameter>; 3],
    pw15: [(Parameter, Parameter); 3],
    pw16: [[Parameter; 3]; 3],
    pw17: [Vec<Parameter>; 3],
    pw18: Vec<Option<Parameter>>,
    pw19: Vec<(Parameter, Parameter)>,
    pw20: Vec<[Parameter; 3]>,
    pw21: Vec<Vec<Parameter>>,
}

impl Model1 {
    pub fn new() -> Self {
        Model1 {
            pw1: Parameter::new(),
            pw2: Some(Parameter::new()),
            pw3: (Parameter::new(), Parameter::new()),
            pw4: [Parameter::new(), Parameter::new(), Parameter::new()],
            pw5: (0..4).map(|_| Parameter::new()).collect(),
            pw6: Some(Some(Parameter::new())),
            pw7: Some((Parameter::new(), Parameter::new())),
            pw8: Some([Parameter::new(), Parameter::new(), Parameter::new()]),
            pw9: Some((0..4).map(|_| Parameter::new()).collect()),
            pw10: (Some(Parameter::new()), Some(Parameter::new())),
            pw11: (
                (Parameter::new(), Parameter::new()),
                (Parameter::new(), Parameter::new()),
            ),
            pw12: (
                [Parameter::new(), Parameter::new(), Parameter::new()],
                [Parameter::new(), Parameter::new(), Parameter::new()],
            ),
            pw13: (
                (0..4).map(|_| Parameter::new()).collect(),
                (0..4).map(|_| Parameter::new()).collect(),
            ),
            pw14: [
                Some(Parameter::new()),
                Some(Parameter::new()),
                Some(Parameter::new()),
            ],
            pw15: [
                (Parameter::new(), Parameter::new()),
                (Parameter::new(), Parameter::new()),
                (Parameter::new(), Parameter::new()),
            ],
            pw16: [
                [Parameter::new(), Parameter::new(), Parameter::new()],
                [Parameter::new(), Parameter::new(), Parameter::new()],
                [Parameter::new(), Parameter::new(), Parameter::new()],
            ],
            pw17: [
                (0..4).map(|_| Parameter::new()).collect(),
                (0..4).map(|_| Parameter::new()).collect(),
                (0..4).map(|_| Parameter::new()).collect(),
            ],
            pw18: (0..4).map(|_| Some(Parameter::new())).collect(),
            pw19: (0..4)
                .map(|_| (Parameter::new(), Parameter::new()))
                .collect(),
            pw20: (0..4)
                .map(|_| [Parameter::new(), Parameter::new(), Parameter::new()])
                .collect(),
            pw21: (0..4)
                .map(|_| (0..4).map(|_| Parameter::new()).collect())
                .collect(),
        }
    }
}

#[derive(Model)]
struct Model2(
    Parameter,
    Option<Parameter>,
    (Parameter, Parameter),
    [Parameter; 3],
    Vec<Parameter>,
);

impl Model2 {
    pub fn new() -> Self {
        Model2(
            Parameter::new(),
            Some(Parameter::new()),
            (Parameter::new(), Parameter::new()),
            [Parameter::new(), Parameter::new(), Parameter::new()],
            (0..4).map(|_| Parameter::new()).collect(),
        )
    }
}

#[derive(Model)]
struct Model3;

#[derive(Model)]
enum Model4 {
    Variant1,
    Variant2(i32),
    Variant3(Parameter),
    Variant4((Parameter, i32, Parameter)),
    Variant5(Parameter, i32, Parameter),
    Variant6 {
        pw1: Parameter,
    },
    Variant7 {
        pw1: (Parameter, i32, Parameter),
    },
    Variant8 {
        pw1: Parameter,
        int_val: i32,
        pw2: Parameter,
    },
}

#[test]
fn derive_named_struct_test() {
    let mut m = Model1::new();
    m.register_parameters();
    assert!(m.get_parameter("pw1").is_some());

    assert!(m.get_parameter("pw2").is_some());
    assert!(m.get_parameter("pw3.0").is_some());
    assert!(m.get_parameter("pw3.1").is_some());
    assert!(m.get_parameter("pw3.2").is_none());
    for (i, _) in m.pw4.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw4.{}", i)).is_some());
    }
    assert!(m.get_parameter(&format!("pw4.{}", m.pw4.len())).is_none());
    for (i, _) in m.pw5.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw5.{}", i)).is_some());
    }
    assert!(m.get_parameter(&format!("pw5.{}", m.pw5.len())).is_none());

    assert!(m.get_parameter("pw6").is_some());
    assert!(m.get_parameter("pw7.0").is_some());
    assert!(m.get_parameter("pw7.1").is_some());
    assert!(m.get_parameter("pw7.2").is_none());
    let pw8 = m.pw8.as_ref().unwrap();
    for (i, _) in pw8.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw8.{}", i)).is_some());
    }
    assert!(m.get_parameter(&format!("pw8.{}", pw8.len())).is_none());
    let pw9 = m.pw9.as_ref().unwrap();
    for (i, _) in pw9.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw9.{}", i)).is_some());
    }
    assert!(m.get_parameter(&format!("pw9.{}", pw9.len())).is_none());

    assert!(m.get_parameter("pw10.0").is_some());
    assert!(m.get_parameter("pw10.1").is_some());
    assert!(m.get_parameter("pw10.2").is_none());
    assert!(m.get_parameter("pw11.0.0").is_some());
    assert!(m.get_parameter("pw11.0.1").is_some());
    assert!(m.get_parameter("pw11.0.2").is_none());
    assert!(m.get_parameter("pw11.1.0").is_some());
    assert!(m.get_parameter("pw11.1.1").is_some());
    assert!(m.get_parameter("pw11.1.2").is_none());
    assert!(m.get_parameter("pw11.2.0").is_none());
    for (i, _) in m.pw12.0.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw12.0.{}", i)).is_some());
    }
    assert!(
        m.get_parameter(&format!("pw12.0.{}", m.pw12.0.len()))
            .is_none()
    );
    for (i, _) in m.pw12.1.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw12.1.{}", i)).is_some());
    }
    assert!(
        m.get_parameter(&format!("pw12.1.{}", m.pw12.1.len()))
            .is_none()
    );
    for (i, _) in m.pw13.0.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw13.0.{}", i)).is_some());
    }
    assert!(
        m.get_parameter(&format!("pw13.0.{}", m.pw13.0.len()))
            .is_none()
    );
    for (i, _) in m.pw13.1.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw13.1.{}", i)).is_some());
    }
    assert!(
        m.get_parameter(&format!("pw13.1.{}", m.pw13.1.len()))
            .is_none()
    );

    for (i, _) in m.pw14.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw14.{}", i)).is_some());
    }
    for (i, _) in m.pw15.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw15.{}.0", i)).is_some());
        assert!(m.get_parameter(&format!("pw15.{}.1", i)).is_some());
        assert!(m.get_parameter(&format!("pw15.{}.2", i)).is_none());
    }
    for (i, p) in m.pw16.iter().enumerate() {
        for (j, _) in p.iter().enumerate() {
            assert!(m.get_parameter(&format!("pw16.{}.{}", i, j)).is_some());
        }
        assert!(
            m.get_parameter(&format!("pw16.{}.{}", i, p.len()))
                .is_none()
        );
    }
    for (i, p) in m.pw17.iter().enumerate() {
        for (j, _) in p.iter().enumerate() {
            assert!(m.get_parameter(&format!("pw17.{}.{}", i, j)).is_some());
        }
        assert!(
            m.get_parameter(&format!("pw17.{}.{}", i, p.len()))
                .is_none()
        );
    }

    for (i, _) in m.pw18.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw18.{}", i)).is_some());
    }
    for (i, _) in m.pw19.iter().enumerate() {
        assert!(m.get_parameter(&format!("pw19.{}.0", i)).is_some());
        assert!(m.get_parameter(&format!("pw19.{}.1", i)).is_some());
        assert!(m.get_parameter(&format!("pw19.{}.2", i)).is_none());
    }
    for (i, p) in m.pw20.iter().enumerate() {
        for (j, _) in p.iter().enumerate() {
            assert!(m.get_parameter(&format!("pw20.{}.{}", i, j)).is_some());
        }
        assert!(
            m.get_parameter(&format!("pw20.{}.{}", i, p.len()))
                .is_none()
        );
    }
    for (i, p) in m.pw21.iter().enumerate() {
        for (j, _) in p.iter().enumerate() {
            assert!(m.get_parameter(&format!("pw21.{}.{}", i, j)).is_some());
        }
        assert!(
            m.get_parameter(&format!("pw21.{}.{}", i, p.len()))
                .is_none()
        );
    }
}

#[test]
fn derive_unnamed_struct_test() {
    let mut m = Model2::new();
    m.register_parameters();
    assert!(m.get_parameter("0").is_some());
    assert!(m.get_parameter("1").is_some());
    assert!(m.get_parameter("2.0").is_some());
    assert!(m.get_parameter("2.1").is_some());
    assert!(m.get_parameter("2.2").is_none());
    for (i, _) in m.3.iter().enumerate() {
        assert!(m.get_parameter(&format!("3.{}", i)).is_some());
    }
    assert!(m.get_parameter(&format!("3.{}", m.3.len())).is_none());
    for (i, _) in m.4.iter().enumerate() {
        assert!(m.get_parameter(&format!("4.{}", i)).is_some());
    }
    assert!(m.get_parameter(&format!("4.{}", m.4.len())).is_none());
}

#[test]
fn derive_unit_struct_test() {
    let mut m = Model3;
    m.register_parameters();
}

#[test]
fn derive_enum_test() {
    {
        let mut m = Model4::Variant1;
        m.register_parameters();
    }
    {
        let mut m = Model4::Variant2(1);
        m.register_parameters();
        assert!(m.get_parameter("pw1").is_none());
    }
    {
        let mut m = Model4::Variant3(Parameter::new());
        m.register_parameters();
        assert!(m.get_parameter("Variant3.0").is_some());
    }
    {
        let mut m = Model4::Variant4((Parameter::new(), 1, Parameter::new()));
        m.register_parameters();
        assert!(m.get_parameter("Variant4.0.0").is_some());
        assert!(m.get_parameter("Variant4.0.1").is_none());
        assert!(m.get_parameter("Variant4.0.2").is_some());
    }
    {
        let mut m = Model4::Variant5(Parameter::new(), 1, Parameter::new());
        m.register_parameters();
        assert!(m.get_parameter("Variant5.0").is_some());
        assert!(m.get_parameter("Variant5.1").is_none());
        assert!(m.get_parameter("Variant5.2").is_some());
    }
    {
        let mut m = Model4::Variant6 {
            pw1: Parameter::new(),
        };
        m.register_parameters();
        assert!(m.get_parameter("Variant6.pw1").is_some());
    }
    {
        let mut m = Model4::Variant7 {
            pw1: (Parameter::new(), 1, Parameter::new()),
        };
        m.register_parameters();
        assert!(m.get_parameter("Variant7.pw1.0").is_some());
        assert!(m.get_parameter("Variant7.pw1.1").is_none());
        assert!(m.get_parameter("Variant7.pw1.2").is_some());
    }
    {
        let mut m = Model4::Variant8 {
            pw1: Parameter::new(),
            int_val: 1,
            pw2: Parameter::new(),
        };
        m.register_parameters();
        assert!(m.get_parameter("Variant8.pw1").is_some());
        assert!(m.get_parameter("Variant8.int_val").is_none());
        assert!(m.get_parameter("Variant8.pw2").is_some());
    }
}
