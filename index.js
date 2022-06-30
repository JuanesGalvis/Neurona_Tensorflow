const classifier = knnClassifier.create()
const webcamElement = document.getElementById("webcam")

let SonrisaImgs = Array.from(document.querySelectorAll('.sonrisa'));
let SorpresaImgs = Array.from(document.querySelectorAll('.sorpresa'));
let LenguaImgs = Array.from(document.querySelectorAll('.triste'));

let loading = document.getElementById('loading');

let net

async function app() {
  console.log("Loading mobilnet...")

  net = await mobilenet.load()

  console.log("Loaded model")
  loading.style.display = 'none';
  window.scrollTo(0, 0);
  
  const webcam = await tf.data.webcam(webcamElement)
  
  const addExample = async (classId) => {
    
    const img = await webcam.capture()
    const activation = net.infer(img, true)
    classifier.addExample(activation, classId)
    img.dispose()
  }
  
  document.getElementById("sonrisa").addEventListener("click", () => addExample(0))
  document.getElementById("sorpresa").addEventListener("click", () => addExample(1))
  document.getElementById("triste").addEventListener("click", () => addExample(2))

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
  LenguaImgs.map((elementHTML) => {
    const img = tf.browser.fromPixels(elementHTML);
    const logits0 = net.infer(img, true);
    classifier.addExample(logits0, 2);
  })

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