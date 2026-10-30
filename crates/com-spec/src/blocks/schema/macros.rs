//! The macros that declare parameters: `block_params!` (a block or a group of
//! parameters) and `json_param!` (a parameter of several JSON shapes).

/// A parameter whose value has several JSON shapes (a name or a schedule, a number or a
/// list), kept as given, with its JSON Schema: `json_param! { /// DOC NAME, SCHEMA }`.
macro_rules! json_param {
    ($(#[$m:meta])* $name:ident, $schema:expr $(,)?) => {
        $(#[$m])*
        #[derive(Clone, Debug, serde::Deserialize)]
        #[serde(transparent)]
        pub(crate) struct $name(pub(crate) serde_json::Value);

        impl $crate::blocks::schema::ParamType for $name {
            fn schema() -> serde_json::Value {
                $schema
            }
        }
    };
}

/// Declare the parameters of a block: the structure, its [`Params`](crate::blocks::schema::Params) parser and its
/// documentation (see the [module documentation](crate::blocks::schema)).
///
/// ```text
/// block_params! {
///     [pub] struct NAME {
///         FIELD: TYPE [DEFAULT] ["UNIT"] [one_of [VALUE, …] | one_of (CONST)] [> B] [>= B] [<= B] [< B], "MEANING";
///         FIELD: GROUP [flatten], "MEANING";
///         …
///     }
/// }
/// ```
///
/// `DEFAULT` is `required`, `absent`, `absent "TEXT"`, a value or `VALUE => "TEXT"`; the
/// type of a field implements [`ParamType`](crate::blocks::schema::ParamType) (its JSON Schema) and `Deserialize`. The
/// accepted values are a list of literals or, in parentheses, a constant array of
/// strings kept with the type they name (`one_of (DetectorKind::NAMES)`). A `[flatten]`
/// field is a group of parameters declared once with `block_params!` and shared by
/// several blocks: its parameters are those of the block, at the place of the field,
/// and a name that neither the block nor the group declares is still rejected.
///
/// ```
/// use com_spec::blocks::Params;
/// com_spec::block_params! {
///     struct Decoder {
///         iterations: usize [50] >= 1, "decoding iterations at most";
///     }
/// }
/// com_spec::block_params! {
///     struct Code {
///         n: usize [required], "length";
///         decoder: Decoder [flatten], "the decoder";
///     }
/// }
/// let c = Code::parse("code", serde_json::json!({"n": 8, "iterations": 5})).unwrap();
/// assert_eq!((c.n, c.decoder.iterations), (8, 5));
/// assert_eq!(Code::params().iter().map(|p| p.name).collect::<Vec<_>>(), ["n", "iterations"]);
/// assert!(Code::parse("code", serde_json::json!({"n": 8, "rule": "min_sum"})).is_err());
/// ```
#[macro_export]
macro_rules! block_params {
    (@default required) => { $crate::blocks::schema::ParamDefault::Required };
    (@default absent) => { $crate::blocks::schema::ParamDefault::Absent("none") };
    (@default absent $text:literal) => { $crate::blocks::schema::ParamDefault::Absent($text) };
    (@default $v:expr => $text:literal) => {
        $crate::blocks::schema::ParamDefault::Value($crate::__macro::json!($v), $text.to_string())
    };
    (@default $v:expr) => {
        $crate::blocks::schema::ParamDefault::Value(
            $crate::__macro::json!($v),
            $crate::blocks::schema::shown(stringify!($v)),
        )
    };
    (@or $d:literal) => { $d };
    (@or $d:literal $v:literal) => { $v };
    (@bound) => { None };
    (@bound $b:literal) => { Some(f64::from($b)) };
    (@choices) => { &[] };
    (@choices [$($c:literal),+ $(,)?]) => { &[$($c),+] };
    (@choices ($c:expr)) => { &$c };
    (@info $v:ident, $field:ident, $ty:ty, [flatten], $($rest:tt)*) => {
        $v.extend(<$ty as $crate::blocks::schema::Params>::params());
    };
    (
        @info $v:ident, $field:ident, $ty:ty, [$($def:tt)+], [$($unit:literal)?], [$($choices:tt)?],
        [$($gt:literal)?], [$($ge:literal)?], [$($le:literal)?], [$($lt:literal)?], $doc:expr
    ) => {
        $v.push($crate::blocks::schema::ParamInfo {
            name: stringify!($field),
            default: $crate::block_params!(@default $($def)+),
            unit: $crate::block_params!(@or "" $($unit)?),
            choices: $crate::block_params!(@choices $($choices)?),
            gt: $crate::block_params!(@bound $($gt)?),
            ge: $crate::block_params!(@bound $($ge)?),
            le: $crate::block_params!(@bound $($le)?),
            lt: $crate::block_params!(@bound $($lt)?),
            doc: $doc,
            schema: <$ty as $crate::blocks::schema::ParamType>::schema(),
            variants: <$ty as $crate::blocks::schema::ParamType>::variants(),
            fields: <$ty as $crate::blocks::schema::ParamType>::fields(),
        });
    };
    (@take $kind:ident, $m:ident, $params:ident, $field:ident, $ty:ty, [flatten]) => {
        <$ty as $crate::blocks::schema::Params>::take_from($kind, $m, $params)?
    };
    (@take $kind:ident, $m:ident, $params:ident, $field:ident, $ty:ty, [$($def:tt)+]) => {
        $crate::blocks::schema::take($kind, $m, $params, stringify!($field))?
    };
    (
        $(#[$sattr:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$fattr:meta])*
                $field:ident : $ty:ty [ $($def:tt)+ ]
                $($unit:literal)?
                $(one_of $choices:tt)?
                $(> $gt:literal)?
                $(>= $ge:literal)?
                $(<= $le:literal)?
                $(< $lt:literal)?
                , $doc:expr ;
            )*
        }
    ) => {
        $(#[$sattr])*
        $vis struct $name {
            $( $(#[$fattr])* $field: $ty, )*
        }

        impl $crate::blocks::schema::Params for $name {
            #[allow(unused_mut, reason = "a block without parameters")]
            fn params() -> Vec<$crate::blocks::schema::ParamInfo> {
                let mut v = Vec::new();
                $(
                    $crate::block_params!(
                        @info v, $field, $ty, [$($def)+], [$($unit)?], [$($choices)?],
                        [$($gt)?], [$($ge)?], [$($le)?], [$($lt)?], $doc
                    );
                )*
                v
            }

            #[allow(unused_variables, reason = "a block without parameters")]
            fn take_from(
                kind: &str,
                m: &mut $crate::__macro::Map<String, $crate::__macro::Json>,
                params: &[$crate::blocks::schema::ParamInfo],
            ) -> $crate::__macro::Result<Self> {
                Ok($name {
                    $( $field: $crate::block_params!(@take kind, m, params, $field, $ty, [$($def)+]), )*
                })
            }
        }
    };
}
