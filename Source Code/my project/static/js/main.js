// Navigation
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
      section.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.add('active');
    
    // Update URL hash
    window.location.hash = sectionId;
  }
  
  // Handle file upload
  let selectedFile = null;
  
  function handleFileSelect(event) {
    selectedFile = event.target.files[0];
    document.getElementById('file-name').textContent = selectedFile ? selectedFile.name : '';
    document.getElementById('submit-btn').disabled = !selectedFile;
  }
  
  async function handleSubmit(event) {
    event.preventDefault();
    
    if (!selectedFile) {
      alert('Please select a file first');
      return;
    }
    
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');
    
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline-block';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (response.ok) {
        // Store results and show results section
        localStorage.setItem('predictionResults', JSON.stringify(data));
        showResults();
        showSection('result');
      } else {
        throw new Error(data.error || 'Failed to process file');
      }
    } catch (error) {
      alert(error.message);
    } finally {
      submitBtn.disabled = false;
      btnText.style.display = 'inline';
      spinner.style.display = 'none';
    }
  }
  
  function showResults() {
    const resultsData = JSON.parse(localStorage.getItem('predictionResults'));
    if (!resultsData) return;
    
    document.getElementById('result-status').textContent = resultsData.result;
    
    const predictionsContainer = document.getElementById('predictions-list');
    predictionsContainer.innerHTML = '';
    
    resultsData.predictions.forEach((pred, index) => {
      const div = document.createElement('div');
      div.className = 'prediction-item';
      div.innerHTML = `
        <span class="label">Row ${index + 1}:</span>
        <span class="${pred === 1 ? 'positive' : 'negative'}">
          ${pred === 1 ? 'Positive' : 'Negative'}
        </span>
      `;
      predictionsContainer.appendChild(div);
    });
  }
  
  // Initialize
  document.addEventListener('DOMContentLoaded', () => {
    // Show initial section based on URL hash or default to home
    const initialSection = window.location.hash.slice(1) || 'home';
    showSection(initialSection);
    
    // Show results if available
    showResults();
  });