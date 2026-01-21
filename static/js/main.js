// // static/js/main.js

// document.addEventListener("DOMContentLoaded", () => {
//   // --- THEME TOGGLE LOGIC (GLOBAL) ---
//   const themeToggle = document.getElementById("theme-toggle");
//   if (themeToggle) {
//     const currentTheme = localStorage.getItem("theme") || "light";
//     document.documentElement.setAttribute("data-theme", currentTheme);

//     themeToggle.addEventListener("click", () => {
//       let newTheme =
//         document.documentElement.getAttribute("data-theme") === "dark"
//           ? "light"
//           : "dark";
//       document.documentElement.setAttribute("data-theme", newTheme);
//       localStorage.setItem("theme", newTheme);
//     });
//   }

//   // --- CHATBOT PAGE LOGIC ---
//   // FIX: Changed the selector to a valid ID on the chatbot page (#chat-box)
//   const chatBox = document.getElementById("chat-box");
//   if (chatBox) {
//     const suggestionCardsContainer =
//       document.getElementById("suggestion-cards");
//     const userInput = document.getElementById("user-input");
//     const sendButton = document.getElementById("send-button");

//     const suggestions = [
//       "What is Tuberculosis?",
//       "How is TB transmitted?",
//       "What are the common symptoms?",
//       "How can TB be prevented?",
//     ];

//     function initChat() {
//       addMessage(
//         "Hello! I'm Aura, your AI assistant. I can provide general information about Tuberculosis. How can I help?",
//         "bot"
//       );
//       renderSuggestionCards();
//     }

//     function renderSuggestionCards() {
//       suggestionCardsContainer.innerHTML = "";
//       suggestions.forEach((text) => {
//         const card = document.createElement("div");
//         card.classList.add("suggestion-card");
//         card.textContent = text;
//         card.addEventListener("click", () => handleSuggestionClick(text));
//         suggestionCardsContainer.appendChild(card);
//       });
//     }

//     function handleSuggestionClick(text) {
//       userInput.value = text;
//       sendMessage();
//     }

//     function addMessage(text, sender) {
//       const messageDiv = document.createElement("div");
//       messageDiv.classList.add("chat-message", sender);

//       const avatar = document.createElement("div");
//       avatar.classList.add("avatar");
//       avatar.innerHTML =
//         sender === "bot"
//           ? '<i class="ph-bold ph-robot"></i>'
//           : '<i class="ph-bold ph-user"></i>';

//       const messageContent = document.createElement("div");
//       messageContent.classList.add("message-content");

//       if (sender === "bot" && text === "...") {
//         messageDiv.classList.add("typing");
//         messageContent.innerHTML = "<p>Aura is typing...</p>";
//       } else {
//         // Use marked.js to render markdown from bot
//         messageContent.innerHTML =
//           sender === "bot" ? marked.parse(text) : `<p>${text}</p>`;
//       }

//       messageDiv.appendChild(avatar);
//       messageDiv.appendChild(messageContent);
//       chatBox.appendChild(messageDiv);
//       chatBox.scrollTop = chatBox.scrollHeight;
//     }

//     function sendMessage() {
//       const message = userInput.value.trim();
//       if (!message) return;

//       addMessage(message, "user");
//       userInput.value = "";
//       suggestionCardsContainer.style.display = "none";
//       addMessage("...", "bot");

//       fetch("/chat", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ message }),
//       })
//         .then((response) => response.json())
//         .then((data) => {
//           const typingIndicator = chatBox.querySelector(".typing");
//           if (typingIndicator) typingIndicator.remove();
//           const reply = data.error ? `Error: ${data.error}` : data.reply;
//           addMessage(reply, "bot");
//         })
//         .catch((error) => {
//           console.error("Error:", error);
//           const typingIndicator = chatBox.querySelector(".typing");
//           if (typingIndicator) typingIndicator.remove();
//           addMessage("Sorry, I encountered an error.", "bot");
//         });
//     }

//     sendButton.addEventListener("click", sendMessage);
//     userInput.addEventListener("keypress", (e) => {
//       if (e.key === "Enter") {
//         sendMessage();
//       }
//     });

//     initChat();
//   }

//   // --- DETECTOR PAGE LOGIC ---
//   const detectorWrapper = document.querySelector(".detector-wrapper");
//   if (detectorWrapper) {
//     const uploadContainer = document.getElementById("upload-container");
//     const resultsContainer = document.getElementById("results-container");
//     const uploadForm = document.getElementById("upload-form");
//     const uploadArea = document.getElementById("upload-area");
//     const fileInput = document.getElementById("file-input");
//     const imagePreview = document.getElementById("image-preview");
//     const predictionText = document.getElementById("prediction-text");
//     const confidenceScore = document.getElementById("confidence-score");
//     const analyzeAnotherBtn = document.getElementById("analyze-another-btn");

//     uploadArea.addEventListener("click", () => fileInput.click());
//     fileInput.addEventListener("change", () => {
//       if (fileInput.files.length > 0) {
//         handleFile(fileInput.files[0]);
//       }
//     });

//     ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
//       uploadArea.addEventListener(eventName, preventDefaults, false);
//     });
//     function preventDefaults(e) {
//       e.preventDefault();
//       e.stopPropagation();
//     }
//     ["dragenter", "dragover"].forEach((eventName) => {
//       uploadArea.addEventListener(
//         eventName,
//         () => uploadArea.classList.add("dragover"),
//         false
//       );
//     });
//     ["dragleave", "drop"].forEach((eventName) => {
//       uploadArea.addEventListener(
//         eventName,
//         () => uploadArea.classList.remove("dragover"),
//         false
//       );
//     });
//     uploadArea.addEventListener("drop", (e) => {
//       if (e.dataTransfer.files.length > 0) {
//         handleFile(e.dataTransfer.files[0]);
//       }
//     });

//     function handleFile(file) {
//       uploadContainer.style.display = "none";
//       resultsContainer.style.display = "flex";

//       const reader = new FileReader();
//       reader.onload = (e) => {
//         imagePreview.src = e.target.result;
//       };
//       reader.readAsDataURL(file);

//       const formData = new FormData();
//       formData.append("file", file);

//       // Show loading state
//       predictionText.textContent = "Analyzing...";
//       predictionText.className = "";
//       confidenceScore.textContent = "---";

//       fetch("/predict", { method: "POST", body: formData })
//         .then((response) => response.json())
//         .then((data) => {
//           if (data.error) {
//             alert(`Error: ${data.error}`);
//             resetDetectorUI();
//             return;
//           }
//           displayDetectorResults(data);
//         })
//         .catch((error) => {
//           console.error("Error:", error);
//           alert("An unexpected error occurred.");
//           resetDetectorUI();
//         });
//     }

//     function displayDetectorResults(data) {
//       const prediction = data.prediction;
//       const confidence = parseFloat(data.confidence);

//       predictionText.textContent = prediction;
//       confidenceScore.textContent = `${(confidence * 100).toFixed(1)}%`;

//       const predictionClass = prediction.toLowerCase();
//       predictionText.className = "";
//       predictionText.classList.add(predictionClass);
//     }

//     function resetDetectorUI() {
//       uploadContainer.style.display = "block";
//       resultsContainer.style.display = "none";
//       uploadForm.reset();
//     }

//     analyzeAnotherBtn.addEventListener("click", resetDetectorUI);
//   }
// });

// static/js/main.js

document.addEventListener("DOMContentLoaded", () => {
  // --- 1. THEME TOGGLE (Works globally) ---
  const themeToggle = document.getElementById("theme-toggle");
  if (themeToggle) {
    const currentTheme = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", currentTheme);

    themeToggle.addEventListener("click", () => {
      let newTheme =
        document.documentElement.getAttribute("data-theme") === "dark"
          ? "light"
          : "dark";
      document.documentElement.setAttribute("data-theme", newTheme);
      localStorage.setItem("theme", newTheme);
    });
  }

  // --- 2. CHATBOT LOGIC (Index Page) ---
  const chatBox = document.getElementById("chat-box");

  // We check if chatBox exists so this code doesn't crash on the Detector page
  if (chatBox) {
    const suggestionCardsContainer =
      document.getElementById("suggestion-cards");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    const suggestions = [
      "What is Tuberculosis?",
      "How is TB transmitted?",
      "What are common symptoms?",
      "How is TB prevented?",
    ];

    function initChat() {
      addMessage(
        "Hello! I'm Aura. I can answer questions about Tuberculosis.",
        "bot"
      );
      renderSuggestionCards();
    }

    function renderSuggestionCards() {
      if (!suggestionCardsContainer) return;
      suggestionCardsContainer.innerHTML = "";
      suggestions.forEach((text) => {
        const card = document.createElement("div");
        card.classList.add("suggestion-card");
        card.textContent = text;
        card.addEventListener("click", () => {
          userInput.value = text;
          sendMessage();
        });
        suggestionCardsContainer.appendChild(card);
      });
    }

    function addMessage(text, sender) {
      const messageDiv = document.createElement("div");
      messageDiv.classList.add("chat-message", sender);

      const avatar = document.createElement("div");
      avatar.classList.add("avatar");
      // Simple Icon logic
      avatar.innerHTML =
        sender === "bot"
          ? '<i class="ph-bold ph-robot"></i>'
          : '<i class="ph-bold ph-user"></i>';

      const messageContent = document.createElement("div");
      messageContent.classList.add("message-content");

      if (sender === "bot" && text === "...") {
        messageDiv.classList.add("typing");
        messageContent.innerHTML = "<p>Aura is typing...</p>";
      } else {
        // FIX: Removed 'marked.parse' to prevent crashes if library is missing.
        // We just wrap text in <p> tags.
        const cleanText = text.replace(/\n/g, "<br>"); // Simple newline handling
        messageContent.innerHTML = `<p>${cleanText}</p>`;
      }

      messageDiv.appendChild(avatar);
      messageDiv.appendChild(messageContent);
      chatBox.appendChild(messageDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    // function sendMessage() {
    //   const message = userInput.value.trim();
    //   if (!message) return;

    //   addMessage(message, "user");
    //   userInput.value = "";
    //   if (suggestionCardsContainer)
    //     suggestionCardsContainer.style.display = "none";
    //   addMessage("...", "bot");

    //   fetch("/chat", {
    //     method: "POST",
    //     headers: { "Content-Type": "application/json" },
    //     body: JSON.stringify({ message }),
    //   })
    //     .then((response) => response.json())
    //     .then((data) => {
    //       const typingIndicator = chatBox.querySelector(".typing");
    //       if (typingIndicator) typingIndicator.remove();

    //       if (data.error) {
    //         addMessage("Error: " + data.error, "bot");
    //       } else {
    //         addMessage(data.reply, "bot");
    //       }
    //     })
    //     .catch((error) => {
    //       console.error("Chat Error:", error);
    //       const typingIndicator = chatBox.querySelector(".typing");
    //       if (typingIndicator) typingIndicator.remove();
    //       addMessage("Network Error: Could not reach the server.", "bot");
    //     });
    // }

    // REPLACE your entire sendMessage function with this:

    function sendMessage() {
      const message = userInput.value.trim();
      if (!message) return;

      addMessage(message, "user");
      userInput.value = "";

      // Hide suggestions if they exist
      if (suggestionCardsContainer)
        suggestionCardsContainer.style.display = "none";

      addMessage("...", "bot");

      // DEBUG: Print exactly where we are sending data
      console.log("Sending message to:", window.location.origin + "/chat");

      // fetch("/chat", {
      //   // <--- ENSURE NO SLASH AT THE END
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" },
      //   body: JSON.stringify({ message: message }),
      // })
      // Calculate the exact URL dynamically
      const chatUrl = window.location.origin + "/chat";

      console.log("Posting to:", chatUrl); // Check console to ensure no double slashes

      fetch(chatUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message }),
      })
        .then((response) => {
          // Check if the server rejected the method (405) or crashed (500/502)
          if (!response.ok) {
            console.error("Server Error Status:", response.status);
            console.error("Server Error Text:", response.statusText);
            throw new Error(
              `Server responded with ${response.status}: ${response.statusText}`
            );
          }
          return response.json();
        })
        .then((data) => {
          // Remove typing indicator
          const typingIndicator = chatBox.querySelector(".typing");
          if (typingIndicator) typingIndicator.remove();

          if (data.error) {
            addMessage("Error: " + data.error, "bot");
          } else {
            addMessage(data.reply, "bot");
          }
        })
        .catch((error) => {
          console.error("Detailed Fetch Error:", error);
          const typingIndicator = chatBox.querySelector(".typing");
          if (typingIndicator) typingIndicator.remove();

          // Show the specific error on screen
          addMessage(`System Error: ${error.message}`, "bot");
        });
    }

    if (sendButton) sendButton.addEventListener("click", sendMessage);
    if (userInput)
      userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
      });

    initChat();
  }

  // --- 3. DETECTOR LOGIC (Detector Page) ---
  const detectorWrapper = document.querySelector(".detector-wrapper");

  // We check if detectorWrapper exists so this code doesn't crash on the Chatbot page
  if (detectorWrapper) {
    const uploadContainer = document.getElementById("upload-container");
    const resultsContainer = document.getElementById("results-container");
    const uploadForm = document.getElementById("upload-form");
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const imagePreview = document.getElementById("image-preview");
    const predictionText = document.getElementById("prediction-text");
    const confidenceScore = document.getElementById("confidence-score");
    const analyzeAnotherBtn = document.getElementById("analyze-another-btn");

    // Click to upload
    if (uploadArea)
      uploadArea.addEventListener("click", () => fileInput.click());

    // Handle file selection
    if (fileInput)
      fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
          handleFile(fileInput.files[0]);
        }
      });

    // Drag and Drop Logic
    if (uploadArea) {
      ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
        uploadArea.addEventListener(
          eventName,
          (e) => {
            e.preventDefault();
            e.stopPropagation();
          },
          false
        );
      });

      uploadArea.addEventListener(
        "dragenter",
        () => uploadArea.classList.add("dragover"),
        false
      );
      uploadArea.addEventListener(
        "dragover",
        () => uploadArea.classList.add("dragover"),
        false
      );
      uploadArea.addEventListener(
        "dragleave",
        () => uploadArea.classList.remove("dragover"),
        false
      );
      uploadArea.addEventListener("drop", (e) => {
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
          handleFile(e.dataTransfer.files[0]);
        }
      });
    }

    function handleFile(file) {
      // UI Update: Show Image, Hide Upload Box
      uploadContainer.style.display = "none";
      resultsContainer.style.display = "flex";

      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
      };
      reader.readAsDataURL(file);

      // Prepare Data
      const formData = new FormData();
      formData.append("file", file); // Must match 'file' in app.py

      // Show "Analyzing..."
      predictionText.textContent = "Analyzing...";
      predictionText.className = "";
      confidenceScore.textContent = "---";

      // Send to Backend
      fetch("/predict", {
        method: "POST",
        body: formData,
      })
        .then(async (response) => {
          // If backend returns 500 or 400, throw error to catch block
          if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Server Error");
          }
          return response.json();
        })
        .then((data) => {
          displayDetectorResults(data);
        })
        .catch((error) => {
          console.error("Detector Error:", error);
          alert("Error: " + error.message);
          resetDetectorUI();
        });
    }

    function displayDetectorResults(data) {
      const prediction = data.prediction;
      const confidence = parseFloat(data.confidence);

      predictionText.textContent = prediction;
      confidenceScore.textContent = `${(confidence * 100).toFixed(1)}%`;

      // Add color class based on result
      predictionText.className = "";
      if (prediction.toLowerCase() === "tuberculosis") {
        predictionText.classList.add("tuberculosis");
      } else {
        predictionText.classList.add("normal");
      }
    }

    function resetDetectorUI() {
      uploadContainer.style.display = "block";
      resultsContainer.style.display = "none";
      uploadForm.reset();
    }

    if (analyzeAnotherBtn)
      analyzeAnotherBtn.addEventListener("click", resetDetectorUI);
  }
});
