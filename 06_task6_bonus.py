from utils import DataLoader
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt

data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_train, y_train)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

background = X_train.sample(50, random_state=42)
explainer = shap.KernelExplainer(svm_model.predict_proba, background)

X_test_sample = X_test.iloc[0:10, :]
shap_values = explainer.shap_values(X_test_sample)

plt.figure()
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

shap.summary_plot(shap_values_to_plot, X_test_sample, show=False)
plt.savefig('task6_kernel_shap_summary.png')

