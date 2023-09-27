    function sendMessage() {
        var userInput = document.getElementById("user-input");
        var chatMessages = document.getElementById("chat-messages");

        // Get the user's input message
        var message = userInput.value.trim();
        if (message === "") {
            return;
        }

        // Display the user's message in the chat window with user-message class
        chatMessages.insertAdjacentHTML('beforeend', `<div class="message user-message">${message}</div>`);
        // Clear the user input field
        userInput.value = "";
        // Make an API call to the server to get the chatbot's response
        fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message: message }),
        })
            .then((response) => response.json())
            .then((data) => {
                // Display the chatbot's response in the chat window with bot-message class
                var botResponse = data.response;
                chatMessages.insertAdjacentHTML('beforeend', `<div class="message bot-message">${botResponse}</div>`);

                // Scroll to the bottom of the chat window to show the latest message
                chatMessages.scrollTop = chatMessages.scrollHeight;
            })
            .catch((error) => {
                console.error("Error:", error);
            });
    }

    function handleKeyPress(event) {
        if (event.keyCode === 13) {
            // If Enter key is pressed, call the sendMessage function
            sendMessage();
        }
    }



let audioChunks = [];
let mediaRecorder;

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    mediaRecorder.start();
    document.getElementById('startRecord').disabled = true;
    document.getElementById('stopRecord').disabled = false;
    } catch (error) {
        console.error("Error accessing the microphone:", error);
    }

}
function stopRecording() {

    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    mediaRecorder.stop();
    document.getElementById('startRecord').disabled = false;
    document.getElementById('stopRecord').disabled = true;

    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

    sendDataToServer(audioBlob).then(message => {
        if (message) {
            chatMessages.insertAdjacentHTML('beforeend', `<div class="message user-message">${message}</div>`);
            // Clear the user input field
            userInput.value = "";

            // Make an API call to the server to get the chatbot's response
            fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: message }),
            })
            .then(response => response.json())
            .then(data => {
                // Display the chatbot's response in the chat window
                var botResponse = data.response;
                chatMessages.insertAdjacentHTML('beforeend', `<div class="message bot-message">${botResponse}</div>`);

                // Scroll to the bottom of the chat window
                chatMessages.scrollTop = chatMessages.scrollHeight;
            })
            .catch(error => {
                console.error("Error:", error);
            });
        }
    });
}

function sendDataToServer(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.wav');
    audioChunks = [];

    return fetch('/transcribe', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.text) {
                document.getElementById('user-input').value = data.text;
                return data.text;
            } else if (data.error) {
                console.error(data.error);
                return null;  // Return null or an empty string if there's an error
            }
        });
}
