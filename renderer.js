document.getElementById('generer').addEventListener('click', () => {
    
    const prompt = document.getElementById('prompt').value;

    fetch('http://localhost:5001/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('resultat').textContent = data.message;
        });
});

document.getElementById('make').addEventListener('click', () => {
    const prompt = document.getElementById('prompt').value;

    fetch('http://localhost:5001/make', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('resultat').textContent = data.message;
        });
});