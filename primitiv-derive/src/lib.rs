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
    let stmts: Vec<quote::Tokens> = variants
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
                        quote! {
                            #name::#variant_ident{#(#fields),*} => {
                                #(#stmts)*
                            }
                        }
                    } else {
                        quote!(#name::#variant_ident(_) => {})
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
                        quote! {
                            #name::#variant_ident(#(#fields),*) => {
                                #(#stmts)*
                            }
                        }
                    } else {
                        quote!(#name::#variant_ident(_) => {})
                    }
                }
                Fields::Unit => quote!(#name::#variant_ident => {}),
            }
        })
        .collect();
    if stmts.len() > 0 {
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
            parse_field(&field_token, &field.ty, is_ref).map(|stmt| {
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
            parse_field(&field_token, &field.ty, is_ref).map(|stmt| {
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

fn parse_field(field: &quote::Tokens, ty: &Type, is_ref: bool) -> Option<quote::Tokens> {
    match FieldType::from_ty(ty) {
        FieldType::Array(sub_type) | FieldType::Vec(sub_type) => {
            parse_field(&quote!(f), sub_type, true).map(|stmt| {
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
                    parse_field(&quote!(#field.#index), sub_type, false).map(|stmt| {
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
        FieldType::Option(sub_type) => parse_field(&quote!(f), sub_type, true).map(|stmt| {
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
