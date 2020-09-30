class RPSDataset {
  //add new eg. to dataset so to keep track of it
  constructor() {
    this.labels = []
  }
  //example is output of prediction for image from truncated mobile net
  addExample(example, label) {
    //for first sample xs is null
    if (this.xs == null) {
      // so set the xs to be the tf.keep for the example,and we push the label into the labels array. 
      this.xs = tf.keep(example); //keep cuz tf.tidy throws tensor but we want this one
      this.labels.push(label);
    } else {
      //For all subsequent samples,we just append the new example to the old. 
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);
      oldX.dispose();
    }
  }
  //takes array and one-hot encode it
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
