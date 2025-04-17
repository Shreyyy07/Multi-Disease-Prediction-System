<h1>🏥 Multi-Purpose Healthcare Prediction System</h1>

<h2> Project Overview</h2>

<p>The <strong>Multi-Purpose Healthcare Prediction System</strong> is a unified web application that enables users to screen for multiple critical diseases — including <strong>Brain Stroke</strong>, <strong>Early Cancer</strong>, <strong>Lung Cancer</strong>, and <strong>Breast Cancer</strong> — from a single platform.<br>
Built using pre-trained machine learning models and an interactive <strong>Streamlit</strong> frontend, this project aims to provide early risk screening tools in a user-friendly and accessible manner.</p>

<p><strong>🔹 Unique Selling Point (USP):</strong><br>
One platform, multiple disease screenings, empowering users to monitor their health proactively.</p>

<hr>

<h2> Features</h2>
<ul>
  <li><strong>Brain Stroke Prediction:</strong> Analyze parameters like age, glucose level, heart disease status, etc.</li>
  <li><strong>Early Cancer Detection:</strong> Symptom-based prediction system for early cancer signs.</li>
  <li><strong>Lung Cancer Prediction:</strong> Screening based on smoking status, chest pain, fatigue, breathlessness.</li>
  <li><strong>Breast Cancer Detection:</strong> Prediction using clinical numerical indicators.</li>
  <li><strong>Instant Reports:</strong> Download health screening reports.</li>
  <li><strong>Risk Visualization:</strong> Graphs and indicators based on symptoms.</li>
</ul>

<hr>

<h2>Tech Stack</h2>
<ul>
  <li><strong>Frontend:</strong> Streamlit</li>
  <li><strong>Backend Models:</strong> Scikit-learn (RandomForest, SVM), Pre-trained .pkl models</li>
  <li><strong>Languages:</strong> Python 3</li>
  <li><strong>Libraries:</strong> Streamlit, scikit-learn, matplotlib, seaborn, pandas, joblib</li>
</ul>

<hr>

<h2> How It Works</h2>
<ol>
  <li>Select the Disease Module (e.g., Stroke, Lung Cancer) from the sidebar.</li>
  <li>Input Personal or Clinical Details into simple forms.</li>
  <li>Receive Prediction Output instantly.</li>
  <li>Download Health Report (for early cancer and other models).</li>
  <li>Visualize Risk Levels (for symptom-based diseases).</li>
</ol>

<hr>

<h2> Project Structure</h2>

<pre>
multi_healthcare_app/
├── app.py
├── models/
│   ├── stroke_model.pkl
│   ├── early_cancer_model.pkl
│   ├── lung_cancer_model.pkl
│   └── breast_cancer_model.pkl
├── README.md
├── requirements.txt
└── venv/ (optional)
</pre>

<hr>

<h2> Getting Started</h2>

<h3>Prerequisites</h3>
<ul>
<li>Python 3.x</li>
<li>pip</li>
</ul>

<h3>Installation</h3>

<pre>
git clone https://github.com/your-username/multi-healthcare-prediction.git
cd multi_healthcare_app

pip install -r requirements.txt

streamlit run app.py
</pre>

<h3>Access the app:</h3>
<p>Open your browser and go to <a href="http://localhost:8501">http://localhost:8501</a></p>

<hr>

<h2> Future Improvements</h2>

<ul>
  <li>Add user authentication and health history saving</li>
  <li>Integrate additional disease models</li>
  <li>Allow users to upload scanned medical reports</li>
  <li>Provide model confidence scores visually</li>
  <li>Add multilingual support</li>
</ul>

<hr>

<h2> License</h2>
<p>This project is intended for educational purposes and not for medical use.</p>
