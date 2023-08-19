const mongoose = require('mongoose');

const eventSchema = new mongoose.Schema({
   name: {
      type: String,
      required: true
   },
   description: String,
   // autres champs...
});

module.exports = mongoose.model('Event', eventSchema);
