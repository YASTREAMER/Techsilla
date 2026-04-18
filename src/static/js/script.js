// Start recording by sending POST request to the Flask backend
function startRecording() {
  // Change the button text to "Started Recording"
  const button = document.querySelector("button[onclick='startRecording()']");
  button.textContent = "Started Recording...";
  button.disabled = true; // Disable the button to prevent multiple clicks

  fetch("/record", {
    method: "POST",
  })
    .then((response) => response.json())
    .then((data) => {
      // After the recording is finished, update the UI
      button.textContent = "Start Recording"; // Reset the button text
      button.disabled = false; // Enable the button again
      document.getElementById("response-text").innerText = data.message;
      document.getElementById("response-container").style.display = "block";
    })
    .catch((error) => {
      console.error("Error:", error);
      // In case of an error, reset the button
      button.textContent = "Start Recording";
      button.disabled = false;
      document.getElementById("response-text").innerText =
        "There was an error with the recording process.";
      document.getElementById("response-container").style.display = "block";
    });
}

function submitText() {
  const userInput = document.getElementById("user-input").value;
  const dropdown = document.getElementById("dropdown");
  const selectedOptionText = dropdown.options[dropdown.selectedIndex].text; // Get the text of the selected option

  if (userInput.trim() === "") {
    alert("Please enter some text!");
    return;
  }

  let formData = new FormData();
  formData.append("text", userInput);
  formData.append("dropdown", selectedOptionText); // Append the selected option text instead of value

  fetch("/submit", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      // After successful submission, get the user input and dropdown value and redirect to /question with query parameters
      window.location.href = `/question?user_input=${encodeURIComponent(userInput)}&dropdown_value=${encodeURIComponent(selectedOptionText)}`;
    })
    .catch((error) => console.error("Error:", error));
}
