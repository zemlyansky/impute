const impute = require('./index')
const make = require('mkdata')

const [Xt, yt] = make.friedman1({ nSamples: 500 })
const Xcomplete = Xt.map((xt, i) => {
  xt[0] = +(xt[0] > 0.5)
  xt[1] = 'class' + xt[1].toFixed(1).slice(2)
  return xt.concat(yt[i])
})
console.log('First rows:', Xcomplete.slice(0, 5))
const Xmiss = Xcomplete.map(
  row => row.map(v => Math.random() < 0.5 ? null : v)
)

const res = impute(Xmiss, { verbose: true, columns: [1]})
console.log('Data:', Xcomplete.slice(0, 10).map(row => row[1]), Xmiss.slice(0, 10).map(row => row[1]), res.data.slice(0, 10).map(row => row[1]))
console.log('Results:', res.importanceMatrix)
