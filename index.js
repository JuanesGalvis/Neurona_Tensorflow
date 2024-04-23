const classifier = knnClassifier.create()
const webcamElement = document.getElementById("webcam")

let SonrisaImgs = Array.from(document.querySelectorAll('.sonrisa'));
let SorpresaImgs = Array.from(document.querySelectorAll('.sorpresa'));
let TristezaImgs = Array.from(document.querySelectorAll('.triste'));

let loading = document.getElementById('loading');

let net

async function app() {
  console.log("Loading mobilnet...")
  // Crear modelo
  net = await mobilenet.load()

  console.log("Loaded model")
  loading.style.display = 'none';
  window.scrollTo(0, 0);
  // Capturar información desde la web
  const webcam = await tf.data.webcam(webcamElement)
  
  const addExample = async (classId) => {
    // Capturar desde webcam
    const img = await webcam.capture();
    console.log(img);
    // Función de activación
    const activation = net.infer(img, true)
    // Agregar imagen a una categoria (0,1,2)
    classifier.addExample(activation, classId)
    // Liberar memoria
    img.dispose()
  }
  
  // Evento click a los botones para entrenar
  document.getElementById("sonrisa").addEventListener("click", () => addExample(0))
  document.getElementById("sorpresa").addEventListener("click", () => addExample(1))
  document.getElementById("triste").addEventListener("click", () => addExample(2))

  /* ...  */

  // ENTRENAMIENTO INICIAL
  SonrisaImgs.map((elementHTML) => {
    const img = tf.browser.fromPixels(elementHTML);
    const logits0 = net.infer(img, true);
    classifier.addExample(logits0, 0);
  })
  SorpresaImgs.map((elementHTML) => {
    const img = tf.browser.fromPixels(elementHTML);
    const logits0 = net.infer(img, true);
    classifier.addExample(logits0, 1);
  })
  TristezaImgs.map((elementHTML) => {
    const img = tf.browser.fromPixels(elementHTML);
    const logits0 = net.infer(img, true);
    classifier.addExample(logits0, 2);
  })

  navigator.mediaDevices.enumerateDevices()
  .then(devices => {
    const cameras = devices.filter(device => device.kind === 'videoinput');
    // Imprime información sobre las cámaras disponibles en la consola
    console.log(cameras);

    // Puedes seleccionar la cámara que desees, por ejemplo, la tercera cámara (índice 2)
    const selectedCamera = cameras[0]; // Esto es solo un ejemplo, puedes elegir la cámara que desees

    // Solicita acceso a la cámara seleccionada
    return navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: selectedCamera.deviceId
      }
    });
  })
  .then(stream => {
    // Aquí puedes utilizar el stream de la cámara
    const videoElement = document.getElementById('webcam');
    videoElement.srcObject = stream;
    videoElement.play();
  })
  .catch(error => {
    console.error('Error al acceder a la cámara:', error);
  });

  while (true) {
    if (classifier.getNumClasses() > 0) {
      
      const img = await webcam.capture()

      const activation = net.infer(img, "conv_preds")

      const result = await classifier.predictClass(activation)

      const classes = ["😃", "😮", "🙁"]

      document.getElementById("console").innerText = `
                ${classes[result.label]}\n
                ${(result.confidences[result.label]*100).toFixed(2)+"%"}
            `

      img.dispose()
    }

    await tf.nextFrame()
  }
}

app()