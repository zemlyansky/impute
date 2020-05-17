const { RandomForestRegressor, RandomForestClassifier } = require('random-forest')
const LinReg = require('ml-regression-multivariate-linear')
const LogRegRaw = require('ml-logistic-regression')
// const importance = require('../importance/index')
const importance = require('importance')
const { Matrix } = require('ml-matrix')
const { detectType } = require('random-forest/src/util')

class LogReg {
  constructor (opts) {
    this.model = new LogRegRaw(opts)
  }

  train (X, y) {
    this.model.train(new Matrix(X), Matrix.columnVector(y))
    return this
  }

  predict (X) {
    return this.model.predict(new Matrix(X))
  }
}

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

function encode (X, types) {
  const Xenc = X.map(row => row.slice(0))
  const encodings = []
  for (const ci in types) {
    if (types[ci] === 'classification') {
      const col = X.map(row => row[ci])
      const uniq = Array.from(new Set(col))
      encodings.push(uniq)
      col.forEach((v, ri) => {
        Xenc[ri][ci] = uniq.indexOf(v)
      })
    } else {
      encodings.push(null)
    }
  }
  return [Xenc, encodings]
}

function fillMeans (Xraw, means) {
  const X = JSON.parse(JSON.stringify(Xraw))
  return X.map(row => row.map((v, i) => isMiss(v) ? means[i] : v))
}

const defaults = {
  columns: null,
  maxIter: 1,
  maxDepth: 10,
  maxFeatures: 'auto',
  minSamplesLeaf: 5,
  minInfoGain: 0,
  nEstimators: 100,
  model: 'rf',
  scaleImp: false,
  verbose: false
}

module.exports = function impute (X, opts) {
  const options = Object.assign({}, defaults, opts)
  const log = options.verbose ? console.log : () => {}
  const n = X.length
  const p = X[0].length
  const columns = options.columns || X[0].map((v, i) => i)
  log('Starting imputation... N: %d, P: %d', n, p)
  log('Target columns:', columns)

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

  let encodings
  let Ximp = X.map(row => row.slice(0))
  let Xbase = fillMeans(X, means)
  const Xupd = Xbase.map(row => row.slice(0))
  const imps = []

  if (!options.model) {
    // Mean imputation only
    Ximp = Ximp.map((row, ri) => row.map((v, ci) => columns.includes(ci) ? Xbase[ri][ci] : v))
  } else {
    // Model-based imputation
    // Encode Xbase (mostly for linear models)
    [Xbase, encodings] = encode(Xbase, types)
    for (let iter = 0; iter < options.maxIter; iter++) {
      log('Iteration', iter + 1)
      for (const i of columns) {
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
        let model
        if (options.model === 'rf') {
          model = types[i] === 'regression'
            ? new RandomForestRegressor({
              nEstimators: 100,
              maxDepth: 15
            })
            : new RandomForestClassifier({
              nEstimators: 100,
              maxDepth: 15
            })
          model.train(Xtrain, ytrain)
        } else {
          model = types[i] === 'regression'
            ? new LinReg(Xtrain, ytrain.map(v => [v]))
            : new LogReg({
              numSteps: 1000,
              learningRate: 5e-3
            }).train(Xtrain, ytrain)
        }
        const impKind = types[i] === 'regression'
          ? 'smape'
          : options.model === 'lr' ? 'acc' : 'ce'
        const imp = importance(model, Xtrain, ytrain, { onlyMeans: true, kind: impKind, scale: options.scaleImp, verbose: options.verbose })
        imp.splice(i, 0, null)
        imps.push(imp)

        if (Xpred.length) {
          const ypred = model.predict(Xpred).flat()
          if (ypred.length) {
            ypred.forEach((v, ri) => {
              const res = types[i] === 'regression'
                ? v
                : encodings[i][v]
              Xupd[missing[i][ri]][i] = res
              Ximp[missing[i][ri]][i] = res
            })
          }
        }
      } // *for column
      Xbase = Xupd.map(row => row.slice(0))
    } // *for iteration
  } // *else
  log('Importances:', imps)

  return {
    data: Ximp,
    importanceMatrix: imps,
    missing: missing,
    nMissing: missing.map(m => m.length),
    varTypes: types
  }
}
