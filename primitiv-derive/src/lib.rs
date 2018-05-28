extern crate proc_macro;
extern crate syn;
#[macro_use]
extern crate quote;

use proc_macro::TokenStream;
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::*;

// TODO(chantera): support generics
//
// ```rust
// struct ModelImpl<T>(T);
//
// impl Model for Modelmpl<Parameter> {
//     fn register_parameters(&mut self) {
//         ...
//     }
// }
//
// impl<M: Model> Model for Modelmpl<M> {
//     fn register_parameters(&mut self) {
//         ...
//     }
// }
// ```

#[proc_macro_derive(Model, attributes(primitiv))]
pub fn derive_model(input: TokenStream) -> TokenStream {
    let input: DeriveInput = syn::parse(input).unwrap();
    match expand_derive_model(&input).into() {
        Ok(expanded) => expanded.into(),
        Err(msg) => panic!(msg),
    }
}

fn expand_derive_model(input: &DeriveInput) -> Result<quote::Tokens, &'static str> {
    let ident = &input.ident;
    let name = ident.to_string();
    let generics = input.generics.clone();
    let (_, ty_generics, _) = input.generics.split_for_impl();
    let (impl_generics, _, where_clause) = generics.split_for_impl();

    let impl_body = match input.data {
        Data::Struct(ref data) => impl_body_from_struct(ident, &data.fields),
        Data::Enum(ref data) => impl_body_from_enum(ident, &data.variants),
        Data::Union(_) => {
            return Err("primitiv does not support derive for unions");
        }
    };
    let impl_model = match impl_body {
        Some(body) => quote! {
            impl #impl_generics primitiv::Model for #ident #ty_generics #where_clause {
                fn register_parameters(&mut self) {
                    let handle: *mut _ = self;
                    unsafe {
                        let model = &mut *handle;
                        #body
                    }
                }

                fn identifier(&self) -> u64 {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::Hasher;
                    let mut hasher = DefaultHasher::new();
                    hasher.write(format!("{}-{:p}", #name, self).as_bytes());
                    hasher.finish()
                }
            }
        },
        None => quote! {
            impl #impl_generics primitiv::Model for #ident #ty_generics #where_clause {
                fn register_parameters(&mut self) {}
            }
        },
    };
    let impl_drop = quote! {
        impl #impl_generics Drop for #ident #ty_generics #where_clause {
            fn drop(&mut self) {
                primitiv::Model::invalidate(self);
            }
        }
    };
    Ok(quote! {
        #impl_model

        #impl_drop
    })
}

fn impl_body_from_struct(_name: &Ident, fields: &Fields) -> Option<quote::Tokens> {
    match fields {
        Fields::Named(ref f) => Some(map_fields(&f.named, true, Some(&Ident::from("self")), None)),
        Fields::Unnamed(ref f) => Some(map_fields(
            &f.unnamed,
            false,
            Some(&Ident::from("self")),
            None,
        )),
        Fields::Unit => None,
    }.and_then(|tokens| {
        let stmts: Vec<quote::Tokens> = tokens.into_iter().filter_map(|token| token).collect();
        if stmts.len() > 0 {
            Some(quote!(#(#stmts)*))
        } else {
            None
        }
    })
}

fn impl_body_from_enum(
    name: &Ident,
    variants: &Punctuated<Variant, Comma>,
) -> Option<quote::Tokens> {
    let stmts: Vec<(quote::Tokens, bool)> = variants
        .iter()
        .map(|variant| {
            let variant_ident = &variant.ident;
            let variant_name = variant_ident.to_string();
            match variant.fields {
                Fields::Named(ref f) => {
                    let tokens = map_fields(&f.named, true, None, Some(&variant_name[..]));
                    if tokens.iter().any(|token| token.is_some()) {
                        let mut fields = Vec::with_capacity(f.named.len());
                        let mut stmts = Vec::with_capacity(tokens.len());
                        f.named
                            .iter()
                            .zip(tokens)
                            .for_each(|(field, token)| match token {
                                Some(stmt) => {
                                    let ident = field.ident.as_ref().unwrap();
                                    fields.push(quote!(ref mut #ident));
                                    stmts.push(stmt);
                                }
                                None => {
                                    let ident = field.ident.as_ref().unwrap();
                                    let unused_ident =
                                        Ident::from(&format!("_{}", ident.to_string())[..]);
                                    fields.push(quote!(#ident: ref mut #unused_ident));
                                }
                            });
                        (
                            quote! {
                                #name::#variant_ident{#(#fields),*} => {
                                    #(#stmts)*
                                }
                            },
                            true,
                        )
                    } else {
                        (quote!(#name::#variant_ident{..} => {}), false)
                    }
                }
                Fields::Unnamed(ref f) => {
                    let tokens = map_fields(&f.unnamed, false, None, Some(&variant_name[..]));
                    if tokens.iter().any(|token| token.is_some()) {
                        let mut fields = Vec::with_capacity(f.unnamed.len());
                        let mut stmts = Vec::with_capacity(tokens.len());
                        tokens
                            .iter()
                            .enumerate()
                            .for_each(|(i, token)| match token {
                                Some(stmt) => {
                                    let ident = Ident::from(&format!("attr{}", i)[..]);
                                    fields.push(quote!(ref mut #ident));
                                    stmts.push(stmt);
                                }
                                None => {
                                    let ident = Ident::from(&format!("_attr{}", i)[..]);
                                    fields.push(quote!(ref mut #ident));
                                }
                            });
                        (
                            quote! {
                                #name::#variant_ident(#(#fields),*) => {
                                    #(#stmts)*
                                }
                            },
                            true,
                        )
                    } else {
                        (quote!(#name::#variant_ident(_) => {}), false)
                    }
                }
                Fields::Unit => (quote!(#name::#variant_ident => {}), false),
            }
        })
        .collect();
    if stmts.len() > 0 && stmts.iter().any(|stmt| stmt.1) {
        let stmts: Vec<_> = stmts.into_iter().map(|stmt| stmt.0).collect();
        Some(quote! { match &mut *self {
            #(#stmts),*
        }})
    } else {
        None
    }
}

fn map_fields(
    fields: &Punctuated<Field, Comma>,
    named: bool,
    root_ident: Option<&Ident>,
    root_name: Option<&str>,
) -> Vec<Option<quote::Tokens>> {
    let iter = fields.iter().enumerate();
    let stmts = if named {
        iter.map(|(_i, field)| {
            let field_ident = field.ident.as_ref().unwrap();
            let (field_token, is_ref) = match root_ident {
                Some(ident) => (quote!(#ident.#field_ident), false),
                None => (quote!(#field_ident), true),
            };
            parse_field(&field_token, &field.ty, &parse_attrs(&field.attrs), is_ref).map(|stmt| {
                let mut field_name = field_ident.to_string();
                if let Some(root) = root_name {
                    field_name = format!("{}.{}", root, field_name);
                }
                quote!({
                    let name = #field_name;
                    #stmt
                })
            })
        }).collect()
    } else {
        iter.map(|(i, field)| {
            let (field_token, is_ref) = match root_ident {
                Some(ident) => {
                    let index = Index::from(i);
                    (quote!(#ident.#index), false)
                }
                None => {
                    let field_ident = Ident::from(&format!("attr{}", i)[..]);
                    (quote!(#field_ident), true)
                }
            };
            parse_field(&field_token, &field.ty, &parse_attrs(&field.attrs), is_ref).map(|stmt| {
                let mut field_name = i.to_string();
                if let Some(root) = root_name {
                    field_name = format!("{}.{}", root, field_name);
                }
                quote!({
                    let name = #field_name;
                    #stmt
                })
            })
        }).collect()
    };
    stmts
}

fn parse_field(
    field: &quote::Tokens,
    ty: &Type,
    attrs: &[FieldAttr],
    is_ref: bool,
) -> Option<quote::Tokens> {
    match FieldType::from_ty(ty, attrs) {
        FieldType::Array(sub_type) | FieldType::Vec(sub_type) => {
            parse_field(&quote!(f), sub_type, attrs, true).map(|stmt| {
                quote! {
                    for (i, f) in #field.iter_mut().enumerate() {
                        let name = format!("{}.{}", name, i);
                        #stmt
                    }
                }
            })
        }
        FieldType::Tuple(sub_types) => {
            let stmts: Vec<_> = sub_types
                .iter()
                .enumerate()
                .filter_map(|(i, sub_type)| {
                    let index = Index::from(i);
                    parse_field(&quote!(#field.#index), sub_type, attrs, false).map(|stmt| {
                        quote!({
                            let name = format!("{}.{}", name, #i);
                            #stmt
                        })
                    })
                })
                .collect();
            if stmts.len() > 0 {
                Some(quote!(#(#stmts)*))
            } else {
                None
            }
        }
        FieldType::Option(sub_type) => parse_field(&quote!(f), sub_type, attrs, true).map(|stmt| {
            quote! {
                if let Some(f) = #field.as_mut() {
                    #stmt
                }
            }
        }),
        FieldType::Parameter => {
            if is_ref {
                Some(quote! {
                    model.add_parameter(&name[..], #field);
                })
            } else {
                Some(quote! {
                    model.add_parameter(&name[..], &mut #field);
                })
            }
        }
        FieldType::Model => {
            if is_ref {
                Some(quote! {
                    #field.register_parameters();
                    model.add_submodel(&name[..], #field);
                })
            } else {
                Some(quote! {
                    #field.register_parameters();
                    model.add_submodel(&name[..], &mut #field);
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
    fn from_ty(ty: &'a Type, attrs: &[FieldAttr]) -> Self {
        match ty {
            Type::Array(ref t) => FieldType::Array(&t.elem),
            Type::Tuple(ref t) => FieldType::Tuple(t.elems.iter().collect()),
            Type::Path(ref t) => match t.path.segments.iter().last().unwrap().ident.as_ref() {
                "Vec" => FieldType::Vec(FieldType::generic_subtype(ty).unwrap()),
                "Option" => FieldType::Option(FieldType::generic_subtype(ty).unwrap()),
                "Parameter" => FieldType::Parameter,
                _ => match attrs.last() {
                    Some(FieldAttr::Parameter) => FieldType::Parameter,
                    Some(FieldAttr::Model) => FieldType::Model,
                    None => FieldType::Other,
                },
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

enum FieldAttr {
    Parameter,
    Model,
}

fn parse_attrs(attrs: &[Attribute]) -> Vec<FieldAttr> {
    let iter = attrs
        .iter()
        .filter_map(|attr| {
            let path = &attr.path;
            match quote!(#path).to_string() == "primitiv" {
                true => Some(
                    attr.interpret_meta()
                        .expect(&format!("invalid primitiv syntax: {}", quote!(attr))),
                ),
                false => None,
            }
        })
        .flat_map(|m| match m {
            Meta::List(l) => l.nested,
            tokens => panic!("unsupported syntax: {}", quote!(#tokens).to_string()),
        })
        .map(|m| match m {
            NestedMeta::Meta(m) => m,
            ref tokens => panic!("unsupported syntax: {}", quote!(#tokens).to_string()),
        });
    iter.filter_map(|attr| match attr {
        Meta::Word(ref w) if w == "parameter" => Some(FieldAttr::Parameter),
        Meta::Word(ref w) if w == "submodel" => Some(FieldAttr::Model),
        ref v @ Meta::NameValue(..) | ref v @ Meta::List(..) | ref v @ Meta::Word(..) => {
            panic!("unsupported option: {}", quote!(#v))
        }
    }).collect()
}
