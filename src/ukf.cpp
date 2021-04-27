#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // ukf is not initialized until first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);
  P_(0,0) = 0.001;
  P_(1,1) = 0.01;
  P_(2,2) = 0.09;
  P_(3,3) = 0.009;
  P_(4,4) = 0.2;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);

  // time when the state is true, in us
  time_us_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 4;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 12;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  // Update true state timestamp and compute time elapsed from previous measurement (in seconds)
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Set up state using latest measurement if not initialized
  if (!is_initialized_)
  {
    is_initialized_ = true;
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
    {
      x_.fill(0.);
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
      // std::cout << "x = " << x_ << std::endl; 
      return;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
    {
      auto ro = static_cast<float>(meas_package.raw_measurements_(0));     
      auto phi = static_cast<float>(meas_package.raw_measurements_(1));
      auto ro_dot = static_cast<float>(meas_package.raw_measurements_(2));
      float vx = ro_dot * cos(phi);
      float vy = ro_dot * sin(phi);
      x_(0) = ro * cos(phi);
      x_(1) = ro * sin(phi);
      x_(2) = sqrt(vx * vx + vy * vy);
      x_(3) = phi;
      x_(4) = 0.;
      return;
    }
    else
    {
      throw std::runtime_error("Unknown sensor type");
        // is likely to be an error
    }
  }

  // Predict state
  Prediction(dt);

  // Update state according to sensor type
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
  {
    if (use_laser_) 
    {
      UpdateLidar(meas_package.raw_measurements_);
    }
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
  {
      if (use_radar_)
      {
        UpdateRadar(meas_package.raw_measurements_);
      }
  }
  else
  {
    throw std::runtime_error("Unknown sensor type");
      // is likely to be an error
  } 
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state
  x_aug.head(5) =  x_;
  x_aug.tail(2).fill(0.);

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  // create augmented sigma points
  // set sigma mean point
  Xsig_aug.col(0) = x_aug;
  // set sigma points to the right and left of mean
  for (unsigned int i=0; i<n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_ + n_aug_) * A_aug.col(i); // right
    Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_ + n_aug_) * A_aug.col(i); // left
  }
  
  // predict sigma points
  for (unsigned int i = 0; i<(2*n_aug_+1); i++)
  {
    // get state variables and noises from sigma point 
    double x        = Xsig_aug(0, i);
    double y        = Xsig_aug(1, i);    
    double v        = Xsig_aug(2, i);
    double psi      = Xsig_aug(3, i);
    double psid     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_psidd = Xsig_aug(6, i);

    // initialize new state variables
    double x_n, y_n, v_n, psi_n, psid_n;

    // if yaw rate is not zero
    if (psid != 0)
    {
      // increment x and y using normal formula
      x_n = x + v / psid * (sin(psi + psid * delta_t) - sin(psi));
      y_n = y + v / psid * (-cos(psi + psid * delta_t) + cos(psi));
    }
    else
    {
      // avoid division by zero
      x_n = x + v * cos(psi) * delta_t;
      y_n = x + v * sin(psi) * delta_t;
    }
    // increment remaining state variables
    v_n    = v;
    psi_n  = psi + psid * delta_t;
    psid_n = psid;

    // add noise
    double  dt2 = delta_t*delta_t;
    x_n     = x_n + 0.5 * dt2 * cos(psi) * nu_a;
    y_n     = y_n + 0.5 * dt2 * sin(psi) * nu_a;
    v_n     = v_n + delta_t * nu_a;
    psi_n   = psi_n + 0.5 * dt2 * nu_psidd;
    psid_n  = psid_n + delta_t * nu_psidd;

    // write predicted sigma points into right column
    Xsig_pred_(0,i) = x_n;
    Xsig_pred_(1,i) = y_n;
    Xsig_pred_(2,i) = v_n;
    Xsig_pred_(3,i) = psi_n;
    Xsig_pred_(4,i) = psid_n;
  }
  
  // predict state mean
  x_ = Xsig_pred_ * weights_;

  // predict state covariance matrix
  P_.fill(0.0);
  // iterate over sigma points
  for (unsigned int i = 0; i < 2*n_aug_+1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // normalize psi angle to -pi, pi
    while (x_diff(3) > M_PI)  x_diff(3) -= 2*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(Eigen::VectorXd z) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // measurement matrix
  MatrixXd H = MatrixXd(2, 5);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  // measurement error covariance matrix
  MatrixXd R = MatrixXd(2,2);
  R << std_laspx_ * std_laspx_, 0,
       0, std_laspy_ * std_laspy_;

  VectorXd z_pred = H * x_;     // predicted measurement
  VectorXd y = z - z_pred;      // Prediction error
  MatrixXd Ht = H.transpose();
  
  MatrixXd S = H * P_ * Ht + R; // Innovation matrix
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;   
  MatrixXd K = PHt * Si;        // Kalman gain

  //new estimate
  x_ += (K * y);
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
  P_ = (I - K * H) * P_;

  // compute NIS
  double eps = y.transpose() * Si * y;
  std::cout << "NIS_lidar " << eps << std::endl; 
}

void UKF::UpdateRadar(Eigen::VectorXd z) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  
  // Transform sigma points into measurement space
  for (unsigned int i = 0; i<(2*n_aug_+1); i++)
  {
    // get state variables from sigma point 
    double x        = Xsig_pred_(0, i);
    double y        = Xsig_pred_(1, i);    
    double v        = Xsig_pred_(2, i);
    double psi      = Xsig_pred_(3, i);

    // measurement model
    double rho  = sqrt(x*x + y*y);
    double phi  = atan2(y, x); 
    double rhod = (x * cos(psi) * v + y * sin(psi) * v) / rho;

    // assign to sigma point matrix
    Zsig.col(i) << rho, phi, rhod;
  }
  // calculate mean predicted measurement
  z_pred = Zsig * weights_;
  
  // calculate innovation covariance matrix S
  S.fill(0.0);
  // iterate over sigma points
  for (unsigned int i = 0; i < 2*n_aug_+1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // normalize phi angles to -pi, pi not needed as they are obtained from atan2
    while (z_diff(1) > M_PI)  z_diff(1) -= 2*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2*M_PI;
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  // measurement noise
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<  std_radr_ * std_radr_,  0,  0,
        0,  std_radphi_ * std_radphi_,  0,
        0,  0,  std_radrd_ * std_radrd_;
  
  // add noise to covariance 
  S = S + R;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);  
  // iterate over sigma points
  for (unsigned int i = 0; i < 2*n_aug_+1; i++)
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // normalize psi angle to -pi, pi
    while (x_diff(3) > M_PI)  x_diff(3) -= 2*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2*M_PI;

    // measurement difference
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // normalize phi angles to -pi, pi
    while (z_diff(1) > M_PI)  z_diff(1) -= 2*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2*M_PI;
    
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  MatrixXd Si = S.inverse();
  K = Tc * Si;

  VectorXd y = z - z_pred;      // Prediction error


  // update state mean and covariance matrix
  x_ += K * y;
  P_ -= K * S * K.transpose();

  // compute NIS
  double eps = y.transpose() * Si * y;
  std::cout << "NIS_radar " << eps << std::endl; 
}