const express = require('express');
const axios = require('axios');

const app = express();
const PORT = 3000;

// middleware pour parser les requêtes POST
app.use(express.json());

app.post('/validate', async (req, res) => {
    try {
        const { scriptContent } = req.body;
        
        // Utilisez l'API du validateur W3C ici (ou une autre méthode appropriée)
        // pour valider le script. Note: ceci est pseudocode.
        const response = await axios.post('https://w3c-validator-api-url', scriptContent);
        
        // Renvoyer le résultat au client
        res.json(response.data);
    } catch (error) {
        res.status(500).send('Erreur lors de la validation');
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
