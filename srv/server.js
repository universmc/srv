const express = require('express');
const axios = require('axios');
const mongoose = require('mongoose');

const app = express();

// Middleware pour lire les données au format JSON
app.use(express.json());

// Connexion à MongoDB
mongoose.connect('mongodb://localhost:27017/maBaseDeDonnees', { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('Could not connect to MongoDB', err));

// Exemple de route pour la validation W3C (ceci est un pseudocode, l'API du validateur W3C nécessite une configuration spécifique)
app.post('/validate/fullstack', async (req, res) => {
    try {
        const { scriptContent } = req.body;

        // Faites appel à l'API du validateur W3C ici ou utilisez une autre méthode appropriée.
        const response = await axios.post('https://w3c-validator-api-url', scriptContent);

        // Renvoyer le résultat au client
        res.json(response.data);
    } catch (error) {
        res.status(500).send('Erreur lors de la validation');
    }
});

// Utilisation du contrôleur (assurez-vous que le chemin soit correct)
const eventsController = require('./controller.js');
app.use('/api', eventsController);

// Démarrage du serveur
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
