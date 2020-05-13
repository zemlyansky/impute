const { RandomForestRegressor, RandomForestClassifier } = require('random-forest')
const { detectType } = require('random-forest/src/util')

function isMiss (v) {
  return v === null || v === undefined || v === ''
}

function mean (x) {
  const xfil = x.filter(v => !isMiss(v))
  return xfil.reduce((a, v) => a + v / xfil.length, 0)
}

// From https://stackoverflow.com/questions/32786798/how-to-get-mode-in-array
function countValues (a, result) {
  result = result || {}
  if (a.length === 0) return result
  const head = a.shift()
  if (result[head]) result[head]++
  else result[head] = 1
  return countValues(a, result)
}

function mode (x) {
  const xfil = x.filter(v => !isMiss(v))
  const count = countValues(xfil)
  console.log(count)
  let maxCount = -Infinity
  let maxValue = null
  Object.keys(count).forEach(val => {
    if (count[val] > maxCount) {
      maxCount = count[val]
      maxValue = val
    }
  })
  return maxValue
}

function fillMeans (Xraw, means) {
  const X = JSON.parse(JSON.stringify(Xraw))
  return X.map(row => row.map((v, i) => isMiss(v) ? means[i] : v))
}

const defaults = {
  columns: null,
  maxIter: 1,
  nEstimators: 100,
  maxDepth: 10,
  maxFeatures: 'auto',
  minSamplesLeaf: 5,
  minInfoGain: 0,
  verbose: false,
  scaleImp: false
}

module.exports = function fill (X, opts) {
  const options = Object.assign({}, defaults, opts)
  const log = options.verbose ? console.log : () => {}
  const n = X.length
  const p = X[0].length
  log('Starting imputation... N: %d, P: %d', n, p)

  // TODO: Check X
  // TODO: Remove completely missing columns

  // Initial mean imputation
  const means = []
  const missing = []
  const types = []

  for (let i = 0; i < p; i++) {
    const m = []
    const col = X.map((row, ri) => {
      if (isMiss(row[i])) m.push(ri)
      return row[i]
    })
    const type = detectType(col)
    types.push(type)
    missing.push(m)
    means.push(type === 'regression' ? mean(col) : mode(col))
  }

  log('Number of missing values by column:', missing.map(m => m.length))
  log('Variable types:', types)
  log('Means:', means)

  let Xbase = fillMeans(X, means)
  const Ximp = Xbase.map(row => row.slice(0))
  const imps = []

  for (let iter = 0; iter < options.maxIter; iter++) {
    log('Iteration', iter + 1)
    for (let i = 0; i < p; i++) {
      log(' > Variable', i + 1)
      const ytrain = Xbase
        .map(row => row[i])
        .filter((_, ri) => !missing[i].includes(ri))
      const Xtrain = Xbase
        .map(row => row.filter((_, ci) => ci !== i))
        .filter((_, ri) => !missing[i].includes(ri))
      const Xpred = Xbase
        .map(row => row.filter((_, ci) => ci !== i))
        .filter((_, ri) => missing[i].includes(ri))
      const rf = types[i] === 'regression'
        ? new RandomForestRegressor({
          nEstimators: 100,
          maxDepth: 15
        })
        : new RandomForestClassifier({
          nEstimators: 100,
          maxDepth: 15
        })
      const impKind = types[i] === 'regression'
        ? 'smape'
        : 'ce'
      rf.train(Xtrain, ytrain)
      let imp = rf.getFeatureImportances(Xtrain, ytrain, { onlyMeans: true, kind: impKind })
      if (options.scaleImp) {
        const impMax = Math.max.apply(Math, imp)
        imp = impMax > 0 ? imp.map(v => v / impMax) : imp
      }
      imp.splice(i, 0, null)
      imps.push(imp)

      if (Xpred.length) {
        const ypred = rf.predict(Xpred)
        if (ypred.length) {
          console.log('Impute data:', Xpred, ypred)
          ypred.forEach((v, ri) => {
            Ximp[missing[i][ri]][i] = v
          })
        }
      }
    }
    Xbase = Ximp.map(row => row.slice(0))
  }

  return {
    data: Ximp,
    importanceMatrix: imps,
    missing: missing,
    nMissing: missing.map(m => m.length),
    varTypes: types
  }
}
