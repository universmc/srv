const express = require('express');
const app = express();

app.get('/generate-svg', (req, res) => {
    // Générer le SVG ici et l'envoyer comme réponse
    const svg = `
        <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
            <circle cx="100" cy="100" r="90" fill="none" stroke="black" stroke-width="2" />
            <!-- Ajouter l'emoji hexagone ici -->
        </svg>
    `;

    res.setHeader('Content-Type', 'image/svg+xml');
    res.send(svg);
});

app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});
