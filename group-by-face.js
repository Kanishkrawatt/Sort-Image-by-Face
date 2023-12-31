const express = require("express");
const axios = require("axios");
const faceapi = require("face-api.js");
const http = require("http");
const bodyParser = require("body-parser");
const { Canvas, Image, ImageData, createCanvas } = require("canvas");
const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use(
  bodyParser.urlencoded({
    extended: true,
  })
);

// Configure face-api.js to use a custom canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Load the pre-trained models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromDisk("./models"),
  faceapi.nets.ssdMobilenetv1.loadFromDisk("./models"),
  faceapi.nets.faceLandmark68Net.loadFromDisk("./models"),
  faceapi.nets.faceRecognitionNet.loadFromDisk("./models"),
]).then(startServer);

async function startServer() {
  app.get("/", async (req, res) => {
    res
      .status(200)
      .send(
        "Welcome to the face recognition API! Hit the API with a POST request with an array of image URLs to get the grouped images."
      );
  });
  app.post("/", async (req, res) => {
    try {
      const { imageUrls } = req.body;
      const imagePromises = imageUrls.map(async (imageUrl) => {
        const { data } = await axios.get(imageUrl, {
          responseType: "arraybuffer",
        });

        // Create a canvas and load the image into it
        const canvas = createCanvas(1, 1); // Initialize a canvas
        const img = new Image();
        img.src = `data:image/png;base64,${Buffer.from(data).toString(
          "base64"
        )}`;
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        return { url: imageUrl, canvas };
      });

      const imageObjects = await Promise.all(imagePromises);

      const recognizedFaces = [];
      for (const imageObject of imageObjects) {
        const img = imageObject.canvas;
        const detections = await faceapi
          .detectAllFaces(img)
          .withFaceLandmarks()
          .withFaceDescriptors();

        if (detections.length > 0) {
          const faceDescriptor = detections[0].descriptor;
          recognizedFaces.push({
            url: imageObject.url,
            descriptor: faceDescriptor,
          });
        }
      }
      // Group the images based on face recognition
      const groupedImages = [];
      const processedFaces = new Set();

      for (const face of recognizedFaces) {
        if (!processedFaces.has(face)) {
          const group = recognizedFaces.filter((otherFace) => {
            if (otherFace === face) return false;
            const distance = faceapi.euclideanDistance(
              face.descriptor,
              otherFace.descriptor
            );
            if (distance > 0.6) {
              return true;
            }
            return false;
          });

          const groupUrls = group.map((item) => item.url);
          groupedImages.push({ urls: groupUrls });
          group.forEach((item) => processedFaces.add(item));
        }
      }

      res.json(groupedImages);
    } catch (error) {
      console.error("Error:", error);
      res
        .status(500)
        .json({ error: "An error occurred while processing the images." });
    }
  });

  const server = http.createServer(app);

  function startServer(port) {
    server.listen(port, () => {
      console.log(`Server is running on port ${port}`);
    });

    server.on("error", (err) => {
      if (err.code === "EADDRINUSE") {
        startServer(port + 1);
      }
    });
  }

  startServer(port);
}

module.exports = app;
