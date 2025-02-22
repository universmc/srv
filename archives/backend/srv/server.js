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
// Middleware pour lire les données au format JSON
app.use(express.json());

// Connexion à MongoDB
mongoose.connect('mongodb://localhost:27017/maBaseDeDonnees', { useNewUrlParser: true, useUnifiedTopology: true })
.then(() => console.log('Connected to MongoDB'))
.catch(err => console.error('Could not connect to MongoDB', err));

// Utilisation du contrôleur
const eventsController = require('..controller.js');
app.use('/api', eventsController);

// Démarrage du serveur
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
   console.log(`Server is running on port ${PORT}`);
});
