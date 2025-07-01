require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/send-notification', async (req, res) => {
  const { title, message } = req.body;

  try {
    const response = await fetch("https://onesignal.com/api/v1/notifications", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Basic ${process.env.ONESIGNAL_REST_API_KEY}`
      },
      body: JSON.stringify({
        app_id: "3fb40bfe-9737-4053-baa6-ac454a879911",
        included_segments: ["All"],
        headings: { en: title },
        contents: { en: message }
      })
    });

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error("Push notification failed:", error);
    res.status(500).json({ error: 'Failed to send notification' });
  }
});

app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
