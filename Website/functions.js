function predictPrice() {
    // Collect user input values
    const area = document.getElementById("area").value;
    const bedrooms = document.getElementById("bedrooms").value;
    const bathrooms = document.getElementById("bathrooms").value;
    const stories = document.getElementById("stories").value;
    const mainroad = document.getElementById("mainroad").value;
    const guestroom = document.getElementById("guestroom").value;
    const basement = document.getElementById("basement").value;
    const hotwaterheating = document.getElementById("hotwaterheating").value;
    const airconditioning = document.getElementById("airconditioning").value;
    const parking = document.getElementById("parking").value;
    const prefarea = document.getElementById("prefarea").value;
    const furnishingstatus = document.getElementById("furnishingstatus").value;
    const city = document.getElementById("city").value; // New city input

    // Construct user input data as JSON
    const inputData = {
        area: parseFloat(area),
        bedrooms: parseInt(bedrooms),
        bathrooms: parseInt(bathrooms),
        stories: parseInt(stories),
        mainroad: mainroad,
        guestroom: guestroom,
        basement: basement,
        hotwaterheating: hotwaterheating,
        airconditioning: airconditioning,
        parking: parseInt(parking),
        prefarea: prefarea,
        furnishingstatus: furnishingstatus,
        city: city 
    };

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(inputData)
    })
    .then(response => response.json())
    .then(data => {

        const predictedPrice = data.predicted_price_inr;
        document.getElementById("result").innerHTML = `Predicted House Price: ₹${predictedPrice.toFixed(2)}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("result").innerHTML = 'Error predicting price.';
    });
}
