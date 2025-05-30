document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const spinner = analyzeBtn.querySelector('.spinner-border');
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('audioFile');
        const file = fileInput.files[0];
        
        if (!file) {
            showError('Please select an audio file');
            return;
        }
        
        // Show loading state
        setLoadingState(true);
        hideResults();
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
            } else {
                showResults(data);
            }
        } catch (err) {
            showError('An error occurred while processing the file');
            console.error(err);
        } finally {
            setLoadingState(false);
        }
    });
    
    function setLoadingState(loading) {
        if (loading) {
            analyzeBtn.classList.add('loading');
            analyzeBtn.disabled = true;
            spinner.classList.remove('d-none');
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Analyzing...';
        } else {
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
            spinner.classList.add('d-none');
            analyzeBtn.innerHTML = 'Analyze Emotion';
        }
    }
    
    function showResults(data) {
        // Set predicted emotion
        document.getElementById('predictedEmotion').textContent = 
            capitalizeFirst(data.predicted_emotion);
        
        // Create probabilities list
        const probabilitiesList = document.getElementById('probabilitiesList');
        probabilitiesList.innerHTML = '';
        
        // Sort emotions by probability
        const sortedEmotions = Object.entries(data.probabilities)
            .sort(([,a], [,b]) => b - a);
        
        sortedEmotions.forEach(([emotion, probability]) => {
            const percentage = (probability * 100).toFixed(1);
            const emotionItem = document.createElement('div');
            emotionItem.className = 'emotion-item';
            emotionItem.innerHTML = `
                <span class="emotion-name">${capitalizeFirst(emotion)}</span>
                <span class="emotion-percentage">${percentage}%</span>
            `;
            probabilitiesList.appendChild(emotionItem);
        });
        
        // Show emotion chart
        const chartImg = document.getElementById('emotionChart');
        chartImg.src = data.chart_url;
        
        // Show results
        results.classList.remove('d-none');
        error.classList.add('d-none');
    }
    
    function showError(message) {
        error.textContent = message;
        error.classList.remove('d-none');
        results.classList.add('d-none');
    }
    
    function hideResults() {
        results.classList.add('d-none');
        error.classList.add('d-none');
    }
    
    function capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
});