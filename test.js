// For testing basic functionality of .getCurrentPosition

window.navigator.geolocation.getCurrentPosition=function(success){
    console.log('getCurrentPositionCalled');
    var position = {
        "coords" : {
            "latitude": "37.7749",
            "longitude": "-122.4194",
            "accuracy": "150",
        }
    };
    success(position);
};
function geoSuccess(position) {
    console.log('geoSuccess');
    console.log(window.navigator.geolocation);
    console.log(position.coords.latitude);
};
window.navigator.geolocation.getCurrentPosition(
    geoSuccess
);






