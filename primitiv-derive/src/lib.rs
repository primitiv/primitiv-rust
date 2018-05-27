extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::*;

#[proc_macro_derive(Model)]
pub fn derive_model(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();
    match expand_derive_model(&input).into() {
        Ok(expanded) => expanded.into(),
        Err(msg) => panic!(msg),
    }
}

fn expand_derive_model(input: &DeriveInput) -> Result<quote::Tokens, &'static str> {
    let ident = &input.ident;
    let generics = input.generics.clone();
    let (_, ty_generics, _) = input.generics.split_for_impl();
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let body = match input.data {
        Data::Struct(ref data) => impl_body_from_struct(&data.fields)?,
        Data::Enum(ref data) => impl_body_from_enum(&data.variants)?,
        Data::Union(_) => {
            return Err("primitiv does not support derive for unions");
        }
    };
    let impl_block = quote! {
        impl #impl_generics primitiv::Model for #ident #ty_generics #where_clause {
            fn register_parameters(&mut self) {
                let handle: *mut _ = self;
                unsafe {
                    let model = &mut *handle;
                    #body
                }
            }
        }

        impl #impl_generics Drop for #ident #ty_generics #where_clause {
            fn drop(&mut self) {
                primitiv::Model::invalidate(self);
            }
        }
    };
    Ok(impl_block)
}

fn impl_body_from_struct(fields: &Fields) -> Result<quote::Tokens, &'static str> {
    let body = match fields {
        Fields::Named(ref f) => {
            let mut stmts: Vec<quote::Tokens> = vec![];
            for field in &f.named {
                let field_ident = field.ident.as_ref().unwrap();
                let field_name = field_ident.to_string();
                if let Some(stmt) =
                    parse_field(&field_name, &quote!(self.#field_ident), &field.ty, false)
                {
                    stmts.push(stmt);
                }
            }
            quote!(#(#stmts)*)
        }
        Fields::Unnamed(ref f) => {
            return Err("not implemented"); // TODO(chantera): Implement
        }
        Fields::Unit => quote!(),
    };
    // panic!("body: {:?}", body);
    Ok(body)
}

fn impl_body_from_enum(
    variants: &Punctuated<Variant, Comma>,
) -> Result<quote::Tokens, &'static str> {
    Err("not implemented") // TODO(chantera): Implement
}

fn parse_field(
    name: &str,
    field: &quote::Tokens,
    ty: &Type,
    is_ref: bool,
) -> Option<quote::Tokens> {
    match FieldType::from_ty(ty) {
        FieldType::Array(sub_type) | FieldType::Vec(sub_type) => match FieldType::from_ty(sub_type)
        {
            FieldType::Array(_)
            | FieldType::Vec(_)
            | FieldType::Tuple(_)
            | FieldType::Option(_) => {
                // parse_field(name).map(|stmt| {
                //     quote! {for (i, param) in #field.iter_mut().enumerate() {
                //     }
                //     }
                // })
                // let stmts: Vec<_> = self.pw4.iter_mut().map(|(i, v)| {
                //     model.add_parameter(#name, &mut #field);
                // }).collect();
                // if stmts.len() > 0 {
                //     Some(quote!(#(#stmts)*))
                // } else {
                //     None
                // }
                None
            }
            FieldType::Parameter => Some(quote! {
                for (i, param) in #field.iter_mut().enumerate() {
                    model.add_parameter(&format!("{}.{}", #name, i), param);
                }
            }),
            FieldType::Model => Some(quote! {
                for (i, sub_model) in #field.iter_mut().enumerate() {
                    sub_model.register_parameters();
                    model.add_submodel(&format!("{}.{}", #name, i), sub_model);
                }
            }),
            FieldType::Other => None,
        },
        FieldType::Tuple(sub_types) => {
            let stmts: Vec<_> = sub_types
                .iter()
                .enumerate()
                .filter_map(|(i, sub_type)| {
                    let sub_name = format!("{}.{}", name, i);
                    let index = Index::from(i);
                    parse_field(&sub_name, &quote!(#field.#index), sub_type, false)
                })
                .collect();
            if stmts.len() > 0 {
                Some(quote!(#(#stmts)*))
            } else {
                None
            }
        }
        FieldType::Option(sub_type) => parse_field(name, &quote!(v), sub_type, true).map(|stmt| {
            quote! {
                if let Some(v) = #field.as_mut() {
                    #stmt
                }
            }
        }),
        FieldType::Parameter => {
            if is_ref {
                Some(quote! {
                    model.add_parameter(#name, #field);
                })
            } else {
                Some(quote! {
                    model.add_parameter(#name, &mut #field);
                })
            }
        }
        FieldType::Model => {
            if is_ref {
                Some(quote! {
                    #field.register_parameters();
                    model.add_submodel(#name, #field);
                })
            } else {
                Some(quote! {
                    #field.register_parameters();
                    model.add_submodel(#name, &mut #field);
                })
            }
        }
        FieldType::Other => None,
    }
}

enum FieldType<'a> {
    Array(&'a Type),
    Vec(&'a Type),
    Tuple(Vec<&'a Type>),
    Option(&'a Type),
    Parameter,
    Model,
    Other,
}

impl<'a> FieldType<'a> {
    fn from_ty(ty: &'a Type) -> Self {
        match ty {
            Type::Array(ref t) => FieldType::Array(&t.elem),
            Type::Tuple(ref t) => FieldType::Tuple(t.elems.iter().collect()),
            Type::Path(ref t) => match t.path.segments.iter().last().unwrap().ident.as_ref() {
                "Vec" => FieldType::Vec(FieldType::generic_subtype(ty).unwrap()),
                "Option" => FieldType::Option(FieldType::generic_subtype(ty).unwrap()),
                "Parameter" => FieldType::Parameter,
                "Model" => FieldType::Model,
                _ => FieldType::Other,
            },
            _ => FieldType::Other,
        }
    }

    fn generic_subtype(ty: &Type) -> Option<&Type> {
        match ty {
            Type::Path(ref t) => match t.path.segments.iter().last().unwrap() {
                PathSegment {
                    arguments:
                        PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                            ref args, ..
                        }),
                    ..
                } if args.len() == 1 =>
                {
                    if let GenericArgument::Type(ref t) = args[0] {
                        Some(t)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }
}
