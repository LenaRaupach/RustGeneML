use csv::Reader;
use ndarray::{Array2, Array1};
use ndarray_linalg::Solve; // For solving linear systems
use std::error::Error;

// Load CSV data into ndarray 2D array for features and 1D array for labels
fn load_data(file_path: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    let mut rdr = Reader::from_path(file_path)?;
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.records() {
        let record = result?;
        // Assume columns: ... features ... last column = label (0 or 1)
        let row: Vec<f64> = record.iter()
            .take(record.len() - 1)
            .map(|x| x.parse().unwrap_or(0.0))
            .collect();
        features.extend(row);
        let label: f64 = record[record.len() - 1].parse().unwrap_or(0.0);
        labels.push(label);
    }

    let n_samples = labels.len();
    let n_features = features.len() / n_samples;
    let features_arr = Array2::from_shape_vec((n_samples, n_features), features)?;
    let labels_arr = Array1::from(labels);

    Ok((features_arr, labels_arr))
}

// Train simple logistic regression using normal equation approximation
fn train_logistic_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    // For simplicity, here fit linear regression weights: (X'X)w = X'y
    // Normally logistic regression requires iterative optimization (e.g. gradient descent)
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    let weights = xtx.solve_into(xty)?;
    Ok(weights)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (x, y) = load_data("src/data/pharmaco_data.csv")?;
    let weights = train_logistic_regression(&x, &y)?;
    println!("Model weights: {:?}", weights);
    Ok(())
}
