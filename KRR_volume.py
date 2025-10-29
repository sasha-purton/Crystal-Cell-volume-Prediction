import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
import matplotlib.pyplot as plt
import time


def get_learning_curve(data, model, kf, filename='test', trials=20):
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    X = np.array([np.concatenate([d["xrd"], d["composition"]]) for d in data])
    y = np.array([d["volume"] for d in data])

    length = len(data)
    increment = int(np.floor(length/trials))
    remainder = length%increment

    training_sizes = np.arange(remainder+increment, length+1, increment)
    mean_errors = []
    std_errors = []
    for n in training_sizes:
        mae_scores = cross_val_score(model, X[:n], y[:n], cv=kf, scoring=mae_scorer)
        mean_errors.append(-np.mean(mae_scores))
        std_errors.append(np.std(mae_scores))

    np.savetxt("directory"+ filename + ".txt",
            np.column_stack([training_sizes, mean_errors, std_errors]),
            header="train_size mean_mae std_mae")
    return training_sizes, mean_errors, std_errors


def plot_learning_curve(training_sizes, mean_errors, std_errors, title, filename='test'):
    logN = np.log10(training_sizes)
    logE = np.log10(mean_errors)
    coeffs = np.polyfit(logN, logE, 1)
    alpha = -coeffs[0]
    beta = 10** coeffs[1]

    plt.errorbar(training_sizes, mean_errors, std_errors, fmt='bo')
    plt.plot(training_sizes, beta/training_sizes**alpha, color='red')
    plt.text(min(training_sizes),min(mean_errors), "E = " + str(round(beta, ndigits=2)) + "*N^(-" +str(round(alpha, ndigits=2)) +")")
    plt.title(title)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training Size")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig("directory" + filename + ".png")
    plt.show()


def optimize_hyperparams_KRR(X, y, model, kf, alpha_range=(-3,1), gamma_range=(-3,1), num_points=10):

    param_grid = {
        "alpha": np.logspace(alpha_range[0], alpha_range[1], num_points),  # from 1e-6 to 1e2
        "gamma": np.logspace(gamma_range[0], gamma_range[1], num_points),  # from 1e-3 to 1e3
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kf,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    start_time = time.time()
    search.fit(X, y)
    print("search time:", time.time() - start_time)

    best_alpha = search.best_params_['alpha']
    best_gamma = search.best_params_['gamma']

    print("alpha:", best_alpha, "gamma:", best_gamma)

    # Convert the grid search results into arrays
    results = search.cv_results_

    alphas = np.unique(results['param_alpha'].data)
    gammas = np.unique(results['param_gamma'].data)

    # Extract mean test scores (neg_mean_squared_error)
    scores = -results['mean_test_score']

    # Reshape into a matrix for the heatmap
    scores_matrix = scores.reshape(len(alphas), len(gammas))

    plt.figure(figsize=(8, 6))
    plt.imshow(scores_matrix, origin='lower', aspect='auto',
            extent=[np.log10(gammas.min()), np.log10(gammas.max()),
                    np.log10(alphas.min()), np.log10(alphas.max())],
            cmap='viridis')
    plt.colorbar(label='Mean CV MSE')
    plt.scatter(np.log10(best_gamma),
                np.log10(best_alpha),
                color='red', marker='x', s=100, label='Best')
    plt.xlabel('log10(kernel width)')
    plt.ylabel('log10(regularization)')
    plt.title('KRR Hyperparameter Optimization')
    plt.legend()
    plt.show()
    return best_alpha, best_gamma


TRIALS = 20

# Load the saved dataset
data = joblib.load("cubic_xrd_dataset.pkl")

# Concatenate XRD and composition into one feature vector per material

model = KernelRidge(kernel="rbf", alpha=0.01291549665014884, gamma=0.1291549665014884)

kf = KFold(n_splits=5, shuffle=True, random_state=22)

# grid search hyper parameter space
training_sizes, mean_errors, std_errors = get_learning_curve(data, model, kf, filename='optimized_KRR')
plot_learning_curve(training_sizes, mean_errors, std_errors, title='Learning Curve for Cell Volume Prediction Using KRR',
                     filename='optimized_KRR')

