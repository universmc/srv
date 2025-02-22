const Groq = require('groq-sdk');
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

async function generateResponse(req, res) {
    try {
        const { prompt } = req.body;
        if (!prompt) {
            return res.status(400).json({ error: 'Le champ "prompt" est requis.' });
        }

        const response = await groq.chat.completions.create({
            model: 'gemma2-9b-it',
            messages: [
                { role: 'assistant', content: "welcom @worker_Pibot tu es l'intelligence artificielle au cœur du projet C, intitulé « blog des Séveloppeurs ». Tu résides donc à cette adresse https://wp.univers-mc.cloud wp-admin worldpress, bienvenu dans le code source main.js nodeJs. Dont le thème world wp-content/themes/XordThemes/$ blog world t'es indexé en système version sur GitHub https://github.com/universmc/blog/, Si tout est bien compris, nous allons commencer par les présentations." },
                { role: 'user', content: prompt }
            ],
        });

        const message = response.choices[0].message.content;
        res.json({ message });
    } catch (error) {
        console.error('Erreur lors de la requête GROQ:', error);
        res.status(500).json({ error: 'Erreur lors de la requête à l\'API GROQ.' });
    }
}

async function makeResponse(req, res) {
    try {
        const { prompt } = req.body;
        if (!prompt) {
            return res.status(400).json({ error: 'Le champ "prompt" est requis.' });
        }

        const response = await groq.chat.completions.create({
            model: 'llama3-8b-8192',
            messages: [
                { role: 'assistant', content: "welcom @worker_Pibot tu es l'intelligence artificielle au cœur du projet C, intitulé « blog des Séveloppeurs ». Tu résides donc à cette adresse https://wp.univers-mc.cloud wp-admin worldpress, bienvenu dans le code source main.js nodeJs. Dont le thème world wp-content/themes/XordThemes/$ blog world t'es indexé en système version sur GitHub https://github.com/universmc/blog/, Si tout est bien compris, nous allons commencer par les présentations." },
                { role: 'user', content: prompt }
            ],
        });

        const message = response.choices[0].message.content;
        res.json({ message });
    } catch (error) {
        console.error('Erreur lors de la requête GROQ:', error);
        res.status(500).json({ error: 'Erreur lors de la requête à l\'API GROQ.' });
    }
}

module.exports = { generateResponse, makeResponse };