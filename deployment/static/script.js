async function classifyImage() {
    const inputElement = document.getElementById("imageUpload");
    const file = inputElement.files[0];
    
    const imgElement = document.getElementById("uploadedImage");
    imgElement.src = URL.createObjectURL(file);
    imgElement.style.display = "block";  // Make the image visible

    const formData = new FormData();
    formData.append("file", file);

    // Replace 'http://localhost:5000/predict' with your API endpoint if deployed elsewhere
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById("result").textContent = result.result;
}
