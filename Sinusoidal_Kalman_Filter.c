// Sinusoidal Kalman Filter
// Leonardo Starling
// https://www.linkedin.com/in/leonardo-starling-6119b8a7/
// https://github.com/leonardostarling/sinusoidal-kalman

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Define the state and covariance structures
typedef struct {
    double data[2];
} State;

typedef struct {
    double data[2][2];
} Covariance;

// Forward declaration of the KalmanFilter structure
typedef struct KalmanFilter KalmanFilter;

// Define the KalmanFilter structure with function pointers
struct KalmanFilter {
    State x_hat;
    Covariance P;
    Covariance Q;
    double R;
    double H[2];
    double F[2][2];

    void (*predict)(KalmanFilter* self);
    void (*update)(KalmanFilter* self, double z);
};

// Function prototypes
void predict(KalmanFilter* self);
void update(KalmanFilter* self, double z);
void initKalmanFilter(KalmanFilter* kf, double omega, double dt, Covariance Q, double R);

// Main function
int main() {
    // Define the parameters
    double omega = 2 * M_PI * 60;	// Angular frequency
    double dt = 0.0001;          	// Time step
    int N = 500;              		// Number of steps

    // Process noise covariance
    Covariance Q = {{{3e-5, 0}, {0, 3e-5}}};

    // Measurement noise covariance
    double R = 0.1;

    // Initialize the Kalman filter
    KalmanFilter kf;
    initKalmanFilter(&kf, omega, dt, Q, R);

    // Example sinusoidal signal
    double t[N];
    double true_signal[N];
    double measurements[N];
    double filtered_signal[N];

    for (int i = 0; i < N; i++) {
        t[i] = i * dt;
        true_signal[i] = sin(omega * t[i]);
        measurements[i] = true_signal[i] + 0.25 * ((double) rand() / RAND_MAX - 0.5);
    }

    // Run the Kalman filter
    for (int k = 0; k < N; k++) {
        double z = measurements[k];

        // Prediction step
        kf.predict(&kf);

        // Update step
        kf.update(&kf, z);

        // Store the filtered signal
        filtered_signal[k] = kf.H[0] * kf.x_hat.data[0] + kf.H[1] * kf.x_hat.data[1];
    }

    // Print the results (for demonstration purposes)
    for (int i = 0; i < N; i++) {
        printf("Time: %f, True Signal: %f, Noisy Measurement: %f, Filtered Signal: %f\n", t[i], true_signal[i], measurements[i], filtered_signal[i]);
    }

    return 0;
}

// Initialize the Kalman filter
void initKalmanFilter(KalmanFilter* kf, double omega, double dt, Covariance Q, double R) {
    // Initial state estimate
    kf->x_hat = (State){{0, 1}};

    // Initial covariance estimate
    kf->P = (Covariance){{{1, 0}, {0, 1}}};

    // Process noise covariance
    kf->Q = Q;

    // Measurement noise covariance
    kf->R = R;

    // Measurement matrix
    kf->H[0] = 1;
    kf->H[1] = 0;

    // State transition matrix
    kf->F[0][0] = cos(omega * dt);
    kf->F[0][1] = -sin(omega * dt);
    kf->F[1][0] = sin(omega * dt);
    kf->F[1][1] = cos(omega * dt);

    // Assign function pointers
    kf->predict = predict;
    kf->update = update;
}

// Prediction step function
void predict(KalmanFilter* self) {
    State x_hat_new;

    // State prediction
    x_hat_new.data[0] = self->F[0][0] * self->x_hat.data[0] + self->F[0][1] * self->x_hat.data[1];
    x_hat_new.data[1] = self->F[1][0] * self->x_hat.data[0] + self->F[1][1] * self->x_hat.data[1];
    self->x_hat = x_hat_new;

    // Covariance prediction
    Covariance P_new;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            P_new.data[i][j] = 0;
            for (int k = 0; k < 2; k++) {
                P_new.data[i][j] += self->F[i][k] * self->P.data[k][j];
            }
        }
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            self->P.data[i][j] = 0;
            for (int k = 0; k < 2; k++) {
                self->P.data[i][j] += P_new.data[i][k] * self->F[j][k];
            }
            self->P.data[i][j] += self->Q.data[i][j];
        }
    }
}

// Update step function
void update(KalmanFilter* self, double z) {
    // Calculate Kalman gain
    double PHt[2];
    PHt[0] = self->P.data[0][0] * self->H[0] + self->P.data[0][1] * self->H[1];
    PHt[1] = self->P.data[1][0] * self->H[0] + self->P.data[1][1] * self->H[1];

    double HPHt = self->H[0] * PHt[0] + self->H[1] * PHt[1];
    double K[2];
    K[0] = PHt[0] / (HPHt + self->R);
    K[1] = PHt[1] / (HPHt + self->R);

    // Update state estimate
    double y = z - (self->H[0] * self->x_hat.data[0] + self->H[1] * self->x_hat.data[1]);
    self->x_hat.data[0] += K[0] * y;
    self->x_hat.data[1] += K[1] * y;

    // Update covariance estimate
    Covariance P_new;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            P_new.data[i][j] = self->P.data[i][j] - K[i] * self->H[0] * self->P.data[0][j] - K[i] * self->H[1] * self->P.data[1][j];
        }
    }
    self->P = P_new;
}


