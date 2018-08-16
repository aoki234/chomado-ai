$(function(){

  var getImageSuccess = function(data){
    $("#imageID").attr("src", "/currentimage");
    console.log("succeded");

  };

  var getImageFailure = function(data){
    console.log("No image data");
  };

  var successResult = function(data){
    $("#prediction").text("この写真の人は、 " + data.pred + "さんに似ています!!")
    $("#confidence").text("信頼性: " + data.confidence + "%です!!")
    var req = {
      url: "/currentimage",
      method: "get"
    };

    var promise = $.ajax(req)
    promise.then(getImageSuccess, getImageFailure);
  };
  var failureResult = function(data){
    alert("Didn't work");
  };

  var fileChange = function(evt){
    var fileOb = $("#fileField")[0].files[0];
    var formData = new FormData();
    formData.append("picfile", fileOb);
    var req = {
      url: "/predict",
      method: "post",
      processData: false,
      contentType: false,
      data: formData
    };

    var promise = $.ajax(req);
    promise.then(successResult, failureResult);
  };

  $("#fileField").change(fileChange);
});
