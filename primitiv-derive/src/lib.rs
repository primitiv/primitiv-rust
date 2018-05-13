extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use syn::*;
use syn::punctuated::Punctuated;
use syn::token::Comma;

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
    let impl_block =
        quote! {
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
                let field_name = field.ident.as_ref().unwrap();
                parse_field(&field_name, &field.ty, &mut stmts);
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

fn parse_field(field_name: &Ident, ty: &Type, stmts: &mut Vec<quote::Tokens>) {
    match FieldType::from_ty(ty) {
        FieldType::Array => {
            // TODO(chantera): Implement
            panic!("not implement");
        }
        FieldType::Vec => {
            // TODO(chantera): Implement
            panic!("not implement");
            // let sub_type = sub_type(ty).unwrap();
            // match FieldType::from_ty(sub_type) {
            //     FieldType::Array | FieldType::Vec | FieldType::Tuple | FieldType::Option => {}
            //     FieldType::Parameter => {}
            //     FieldType::Model => {}
            //     FieldType::Other => {}
            //     _ => {}
            // }
        }
        FieldType::Tuple => {
            // TODO(chantera): Implement
            panic!("not implement");
        }
        FieldType::Option => {
            let sub_type = sub_type(ty).unwrap();
            match FieldType::from_ty(sub_type) {
                FieldType::Array | FieldType::Vec | FieldType::Tuple | FieldType::Option => {
                    // TODO(chantera): Implement
                    panic!("not implement");
                    // parse_field(field_name, sub_type, stmts)
                }
                FieldType::Parameter => {
                    let name = field_name.to_string();
                    stmts.push(quote! {
                        if let Some(#field_name) = self.#field_name.as_mut() {
                            model.add_parameter(#name, #field_name);
                        }
                    });
                }
                FieldType::Model => {
                    let name = field_name.to_string();
                    stmts.push(quote! {
                        if let Some(#field_name) = self.#field_name.as_mut() {
                            #field_name.register_parameters();
                            model.add_submodel(#name, #field_name);
                        }
                    });
                }
                FieldType::Other => {}
            }
        }
        FieldType::Parameter => {
            let name = field_name.to_string();
            stmts.push(quote! {
                model.add_parameter(#name, &mut self.#field_name);
            });
        }
        FieldType::Model => {
            let name = field_name.to_string();
            stmts.push(quote! {
                self.#field_name.register_parameters();
                model.add_submodel(#name, &mut self.#field_name);
            });
        }
        FieldType::Other => {}
    }
}

enum FieldType {
    Array,
    Vec,
    Tuple,
    Option,
    Parameter,
    Model,
    Other,
}

impl FieldType {
    fn from_ty(ty: &Type) -> Self {
        match ty {
            Type::Array(ref ty) => FieldType::Array,
            Type::Tuple(ref ty) => FieldType::Tuple,
            Type::Path(ref ty) => {
                match ty.path.segments.iter().last().unwrap().ident.as_ref() {
                    "Vec" => FieldType::Vec,
                    "Option" => FieldType::Option,
                    "Parameter" => FieldType::Parameter,
                    "Model" => FieldType::Model,
                    _ => FieldType::Other,
                }
            }
            _ => FieldType::Other,
        }
    }
}

fn sub_type(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Array(ref ty) => None,
        Type::Tuple(ref ty) => None,
        Type::Path(ref ty) => {
            match ty.path.segments.iter().last().unwrap() {
                PathSegment {
                    arguments: PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                                                      ref args, ..
                                                  }),
                    ..
                } if args.len() == 1 => {
                    if let GenericArgument::Type(ref ty) = args[0] {
                        Some(ty)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}
