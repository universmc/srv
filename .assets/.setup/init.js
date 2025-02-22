const path = require('path');
const fs = require('fs');

const express = require('express');
const app = express();
const port = process.env.PORT || 144;


const { Telegraf } = require('telegraf');
const Groq = require('groq-sdk');
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });


// Configuration de l'application...
const configPath = path.join(__dirname, 'srv/config.json');

// Lancement de l'application
app.listen(port, () => {
  console.log(`BrainStorm is listening on port ${port}`);
});