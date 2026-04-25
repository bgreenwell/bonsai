use anyhow::Result;

/// Metadata extracted from the `model.ini` file inside an H2O MOJO archive.
#[allow(dead_code)]
pub struct ModelMetadata {
    pub h2o_version:   String,
    pub algorithm:     String,   // "gbm" or "drf"
    pub category:      String,
    pub n_trees:       usize,
    pub n_features:    usize,
    pub columns:       Vec<String>,
    pub init_f:        f64,
    pub distribution:  String,
    pub link_function: String,
}

/// Parse the contents of `model.ini` extracted from a MOJO `.zip`.
///
/// The format is INI-like but non-standard: the `[columns]` section contains
/// bare column names (one per line) rather than key=value pairs.
pub fn parse_model_ini(content: &str) -> Result<ModelMetadata> {
    let mut h2o_version   = String::new();
    let mut algorithm     = String::new();
    let mut category      = String::new();
    let mut n_trees       = 0usize;
    let mut n_features    = 0usize;
    let mut columns       = Vec::new();
    let mut init_f        = 0.0f64;
    let mut distribution  = String::new();
    let mut link_function = String::new();

    let mut in_columns_section = false;

    for line in content.lines() {
        let line = line.trim();

        if line == "[columns]" {
            in_columns_section = true;
            continue;
        }

        if line.starts_with('[') {
            in_columns_section = false;
            continue;
        }

        if in_columns_section && !line.is_empty() {
            columns.push(line.to_string());
            continue;
        }

        if let Some((key, value)) = line.split_once('=') {
            let key   = key.trim();
            let value = value.trim();

            match key {
                "h2o_version"  => h2o_version  = value.to_string(),
                "algo"         => algorithm    = value.to_string(),
                "category"     => category     = value.to_string(),
                "n_trees"      => n_trees      = value.parse().unwrap_or(0),
                "n_features"   => n_features   = value.parse().unwrap_or(0),
                "init_f"       => init_f       = value.parse().unwrap_or(0.0),
                "distribution" => distribution = value.to_string(),
                "link_function"=> link_function= value.to_string(),
                _ => {}
            }
        }
    }

    if h2o_version.is_empty() || algorithm.is_empty() {
        anyhow::bail!("model.ini is missing required fields (h2o_version or algo)");
    }

    Ok(ModelMetadata {
        h2o_version,
        algorithm,
        category,
        n_trees,
        n_features,
        columns,
        init_f,
        distribution,
        link_function,
    })
}
