const express = require('express');
const router = express.Router();

// Supposons que vous ayez un modèle, disons "Event" pour gérer vos événements de la Timeline.
const Event = require('models/event');

// Obtenir tous les événements
router.get('/events', async (req, res) => {
    try {
        const events = await Event.find();
        res.json(events);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

// Obtenir un événement spécifique
router.get('/events/:id', async (req, res) => {
    try {
        const event = await Event.findById(req.params.id);
        if (event == null) {
            return res.status(404).json({ message: "Cannot find event" });
        }
        res.json(event);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

// Ajouter un événement
router.post('/events', async (req, res) => {
    const event = new Event({
        name: req.body.name,
        description: req.body.description,
        // autres attributs...
    });

    try {
        const newEvent = await event.save();
        res.status(201).json(newEvent);
    } catch (err) {
        res.status(400).json({ message: err.message });
    }
});

// Modifier un événement
router.patch('/events/:id', async (req, res) => {
    try {
        const event = await Event.findById(req.params.id);
        if (event == null) {
            return res.status(404).json({ message: "Cannot find event" });
        }

        if (req.body.name != null) {
            event.name = req.body.name;
        }
        // autres mises à jour...

        const updatedEvent = await event.save();
        res.json(updatedEvent);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

// Supprimer un événement
router.delete('/events/:id', async (req, res) => {
    try {
        const event = await Event.findById(req.params.id);
        if (event == null) {
            return res.status(404).json({ message: "Cannot find event" });
        }

        await event.remove();
        res.json({ message: "Event deleted" });
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

module.exports = router;
