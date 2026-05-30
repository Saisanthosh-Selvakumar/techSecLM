console.log("TechSecLM Initialized");

async function uploadFile() {

    const fileInput =
        document.getElementById("fileInput");

    if (!fileInput.files.length) {
        alert("Select a file first.");
        return;
    }

    const formData = new FormData();

    formData.append(
        "file",
        fileInput.files[0]
    );

    const response =
        await fetch(
            "http://localhost:5000/upload",
            {
                method: "POST",
                body: formData
            }
        );

    const result =
        await response.json();

    alert(
        `Severity: ${result.severity}
Risk Score: ${result.risk_score}`
    );

    console.log(result);
}

function sendMessage() {
    alert("Chat feature coming soon.");
}