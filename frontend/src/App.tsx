import { useEffect, useMemo, useRef, useState } from 'react'

type Language = 'en' | 'kn'
type Score = { class: string; score: number }
type Prediction = {
  status: 'yakshagana' | 'not_yakshagana' | 'uncertain'
  predicted_class: string | null
  candidate_class: string | null
  confidence: number | null
  dataset_similarity: number
  class_scores: Score[]
  confidence_notice: string
}
type ClassMetric = {
  label: string
  image_count: number
  train_count: number
  validation_count: number
  test_count: number
  precision: number
  recall: number
  f1: number
}
type Metrics = {
  model: string
  counts: { train: number; validation: number; test: number }
  accuracy: number
  classifier_accuracy: number
  uncertain_count: number
  uncertain_rate: number
  novelty_screen: {
    threshold: number
    validation_positive_acceptance_rate: number
    test_positive_rejection_count: number
    test_positive_rejection_rate: number
  }
  uncertainty_thresholds: {
    confidence: number
    margin: number
    target_accepted_accuracy: number
    validation_accepted_accuracy: number
  }
  classes: ClassMetric[]
  confusion_matrix: {
    actual_labels: string[]
    predicted_labels: string[]
    values: number[][]
  }
  non_yakshagana: {
    supported: boolean
    screen_available: boolean
    false_acceptance_rate: number | null
    false_rejection_rate: number
    message: string
  }
  confidence_notice: string
}
type Reference = { label: string; url: string }

const copy = {
  en: {
    identify: 'Identify vesha',
    performance: 'Model performance',
    tradition: 'A living tradition, in every look.',
    intro:
      'Explore the expressive world of Yakshagana. Add a clear, single-performer photo to discover which learned vesha it most resembles.',
    uploadTitle: 'Bring a vesha into focus',
    uploadHint: 'Drop one image here, or choose from your device',
    choose: 'Choose image',
    camera: 'Use camera',
    fileTypes: 'JPG, PNG or WEBP · up to 10 MB',
    crop: 'For group photos, crop to one main performer.',
    identifyButton: 'Identify vesha',
    analysing: 'Reading the details…',
    reset: 'Choose another photo',
    result: 'Your result',
    yak: 'Yakshagana vesha',
    notYak: 'Not a Yakshagana Vesha',
    uncertain: 'Uncertain — try a clearer photo',
    confidence: 'Confidence',
    confidenceNote:
      'Confidence compares only the learned vesha classes; it does not prove that an image is Yakshagana.',
    datasetMismatch:
      'This image does not sufficiently resemble the Yakshagana training images.',
    classScores: 'Class scores',
    comparison: 'A closer comparison',
    yourPhoto: 'Your photo',
    reference: 'Dataset reference',
    topChoices: 'Closest learned vesha styles',
    allScores: 'Scores across all classes',
    validationNotice:
      'A best-effort novelty screen compares photos with the Yakshagana training images. A vesha result also requires at least 85% top-class confidence; that score is not 85% accuracy. The screen can still accept unrelated photos or reject unusual Yakshagana photos, and its non-Yakshagana false-accept rate is unknown.',
    modelBadge: '6 learned classes',
    footer:
      'A learning aid, not an authority on a centuries-old art form.',
    metricAccuracy: 'Model accuracy',
    testImages: 'Held-out test images',
    validationAccuracy: 'Validation accepted accuracy',
    uncertainTest: 'Test results marked uncertain',
    yakImagesRejected: 'Yakshagana test images rejected',
    classPerformance: 'Per-class performance',
    class: 'Class',
    images: 'Images',
    precision: 'Precision',
    recall: 'Recall',
    f1: 'F1 score',
    confusion: 'Confusion matrix',
    actual: 'Actual class',
    predicted: 'Predicted class',
    nonYakTitle: 'Non-Yakshagana rejection',
    nonYakUnavailable:
      'No non-Yakshagana test images are available, so the false-accept rate is unknown. A photo unlike the training images is rejected; a similar-looking unrelated photo may still pass.',
    split: 'Data split',
    train: 'Training',
    validation: 'Validation',
    test: 'Test',
    back: 'Back to identifier',
    loading: 'Loading model information…',
    errorTitle: 'Could not load model information',
    bilingual: 'ಕನ್ನಡ',
    footerAccuracy: 'Confidence is not a guarantee.',
    apiUnavailable:
      'The model service is unavailable. Start the FastAPI server and try again.',
  },
  kn: {
    identify: 'ವೇಷ ಗುರುತಿಸಿ',
    performance: 'ಮಾದರಿಯ ಕಾರ್ಯಕ್ಷಮತೆ',
    tradition: 'ಪ್ರತಿ ವೇಷದಲ್ಲೂ ಜೀವಂತ ಪರಂಪರೆ.',
    intro:
      'ಯಕ್ಷಗಾನದ ಅಭಿವ್ಯಕ್ತಿಪೂರ್ಣ ಲೋಕವನ್ನು ಅನ್ವೇಷಿಸಿ. ಕಲಿತ ವೇಷವನ್ನು ಗುರುತಿಸಲು ಒಬ್ಬ ಕಲಾವಿದರ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಸೇರಿಸಿ.',
    uploadTitle: 'ವೇಷದ ಚಿತ್ರವನ್ನು ಸೇರಿಸಿ',
    uploadHint: 'ಚಿತ್ರವನ್ನು ಇಲ್ಲಿ ಬಿಡಿ ಅಥವಾ ಸಾಧನದಿಂದ ಆಯ್ಕೆಮಾಡಿ',
    choose: 'ಚಿತ್ರ ಆಯ್ಕೆಮಾಡಿ',
    camera: 'ಕ್ಯಾಮೆರಾ ಬಳಸಿ',
    fileTypes: 'JPG, PNG ಅಥವಾ WEBP · ಗರಿಷ್ಠ 10 MB',
    crop: 'ಗುಂಪಿನ ಚಿತ್ರವಿದ್ದರೆ, ಒಬ್ಬ ಮುಖ್ಯ ಕಲಾವಿದರನ್ನು ಮಾತ್ರ ಕ್ರಾಪ್ ಮಾಡಿ.',
    identifyButton: 'ವೇಷ ಗುರುತಿಸಿ',
    analysing: 'ಚಿತ್ರವನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ…',
    reset: 'ಬೇರೆ ಚಿತ್ರ ಆಯ್ಕೆಮಾಡಿ',
    result: 'ಫಲಿತಾಂಶ',
    yak: 'ಯಕ್ಷಗಾನ ವೇಷ',
    notYak: 'ಯಕ್ಷಗಾನ ವೇಷವಲ್ಲ',
    uncertain: 'ಖಚಿತವಿಲ್ಲ — ಸ್ಪಷ್ಟವಾದ ಚಿತ್ರ ಪ್ರಯತ್ನಿಸಿ',
    confidence: 'ವಿಶ್ವಾಸ ಮಟ್ಟ',
    confidenceNote:
      'ವಿಶ್ವಾಸ ಮಟ್ಟವು ಕಲಿತ ವೇಷಗಳ ನಡುವಿನ ಹೋಲಿಕೆ ಮಾತ್ರ; ಚಿತ್ರ ಯಕ್ಷಗಾನವೆಂಬುದಕ್ಕೆ ಖಾತರಿಯಲ್ಲ.',
    datasetMismatch:
      'ಈ ಚಿತ್ರವು ಯಕ್ಷಗಾನ ತರಬೇತಿ ಚಿತ್ರಗಳಿಗೆ ಸಾಕಷ್ಟು ಹೋಲಿಕೆಯಾಗುವುದಿಲ್ಲ.',
    classScores: 'ವರ್ಗದ ಅಂಕಗಳು',
    comparison: 'ಹೋಲಿಕೆ ಮಾಡಿ',
    yourPhoto: 'ನಿಮ್ಮ ಚಿತ್ರ',
    reference: 'ಡೇಟಾಸೆಟ್ ಉಲ್ಲೇಖ ಚಿತ್ರ',
    topChoices: 'ಹತ್ತಿರದ ಕಲಿತ ವೇಷಗಳು',
    allScores: 'ಎಲ್ಲಾ ವರ್ಗಗಳ ಅಂಕಗಳು',
    validationNotice:
      'ಈ ಪ್ರಾಯೋಗಿಕ ಹೊಸತನ ಪರಿಶೀಲಕವು ಚಿತ್ರಗಳನ್ನು ಯಕ್ಷಗಾನ ತರಬೇತಿ ಚಿತ್ರಗಳೊಂದಿಗೆ ಹೋಲಿಸುತ್ತದೆ. ವೇಷದ ಫಲಿತಾಂಶಕ್ಕೆ ಕನಿಷ್ಠ 85% ವರ್ಗ ವಿಶ್ವಾಸ ಅಂಕ ಬೇಕು; ಇದು 85% ನಿಖರತೆ ಎಂದಲ್ಲ. ಸಂಬಂಧವಿಲ್ಲದ ಚಿತ್ರಗಳನ್ನು ಸ್ವೀಕರಿಸಬಹುದು ಅಥವಾ ಅಪರೂಪದ ಯಕ್ಷಗಾನ ಚಿತ್ರಗಳನ್ನು ತಿರಸ್ಕರಿಸಬಹುದು; ಸುಳ್ಳು ಸ್ವೀಕಾರದ ಪ್ರಮಾಣ ತಿಳಿದಿಲ್ಲ.',
    modelBadge: '6 ಕಲಿತ ವರ್ಗಗಳು',
    footer: 'ಶತಮಾನಗಳ ಕಲೆಯ ಅಧ್ಯಯನಕ್ಕೆ ನೆರವು — ಅಂತಿಮ ತೀರ್ಪಲ್ಲ.',
    metricAccuracy: 'ಮಾದರಿಯ ನಿಖರತೆ',
    testImages: 'ಸ್ವತಂತ್ರ ಪರೀಕ್ಷಾ ಚಿತ್ರಗಳು',
    validationAccuracy: 'ಮೌಲ್ಯಮಾಪನದ ಸ್ವೀಕೃತ ನಿಖರತೆ',
    uncertainTest: 'ಅನಿಶ್ಚಿತವೆಂದು ಗುರುತಿಸಿದ ಪರೀಕ್ಷೆಗಳು',
    yakImagesRejected: 'ತಿರಸ್ಕರಿಸಿದ ಯಕ್ಷಗಾನ ಪರೀಕ್ಷಾ ಚಿತ್ರಗಳು',
    classPerformance: 'ವರ್ಗವಾರು ಕಾರ್ಯಕ್ಷಮತೆ',
    class: 'ವರ್ಗ',
    images: 'ಚಿತ್ರಗಳು',
    precision: 'ಪ್ರಿಸಿಷನ್',
    recall: 'ರೀಕಾಲ್',
    f1: 'F1 ಅಂಕ',
    confusion: 'ಗೊಂದಲ ಮ್ಯಾಟ್ರಿಕ್ಸ್',
    actual: 'ನಿಜವಾದ ವರ್ಗ',
    predicted: 'ಊಹಿಸಿದ ವರ್ಗ',
    nonYakTitle: 'ಯಕ್ಷಗಾನವಲ್ಲದ ಚಿತ್ರ ತಿರಸ್ಕಾರ',
    nonYakUnavailable:
      'ಯಕ್ಷಗಾನವಲ್ಲದ ಪರೀಕ್ಷಾ ಚಿತ್ರಗಳಿಲ್ಲ; ಆದ್ದರಿಂದ ಸುಳ್ಳು ಸ್ವೀಕಾರದ ಪ್ರಮಾಣ ತಿಳಿದಿಲ್ಲ. ತರಬೇತಿ ಚಿತ್ರಗಳಿಗೆ ಹೋಲದ ಚಿತ್ರ ತಿರಸ್ಕಾರವಾಗುತ್ತದೆ; ಹೋಲಿಕೆಯ ಚಿತ್ರ ಇನ್ನೂ ಸ್ವೀಕರಿಸಬಹುದು.',
    split: 'ಡೇಟಾ ವಿಭಾಗ',
    train: 'ತರಬೇತಿ',
    validation: 'ಮೌಲ್ಯಮಾಪನ',
    test: 'ಪರೀಕ್ಷೆ',
    back: 'ಗುರುತಿಸುವಿಕೆಗೆ ಹಿಂತಿರುಗಿ',
    loading: 'ಮಾದರಿಯ ಮಾಹಿತಿ ಲೋಡ್ ಆಗುತ್ತಿದೆ…',
    errorTitle: 'ಮಾದರಿಯ ಮಾಹಿತಿ ಲೋಡ್ ಆಗಲಿಲ್ಲ',
    bilingual: 'English',
    footerAccuracy: 'ವಿಶ್ವಾಸ ಮಟ್ಟ ಖಾತರಿಯಲ್ಲ.',
    apiUnavailable:
      'ಮಾದರಿ ಸೇವೆ ಲಭ್ಯವಿಲ್ಲ. FastAPI ಸರ್ವರ್ ಪ್ರಾರಂಭಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
  },
} as const

const kannadaNames: Record<string, string> = {
  Bannadavesha: 'ಬಣ್ಣದ ವೇಷ',
  Mahishasura: 'ಮಹಿಷಾಸುರ',
  'Pundarika veshsa': 'ಪುಂಡರೀಕ ವೇಷ',
  'Raja vesha': 'ರಾಜ ವೇಷ',
  'Stree vesha': 'ಸ್ತ್ರೀ ವೇಷ',
  'Vidhushaka or vidhushaka': 'ವಿದೂಷಕ',
  Uncertain: 'ಅನಿಶ್ಚಿತ',
  'Rejected as Not Yakshagana': 'ಯಕ್ಷಗಾನವಲ್ಲ ಎಂದು ತಿರಸ್ಕರಿಸಲಾಗಿದೆ',
}

function nameFor(label: string, language: Language) {
  return language === 'kn' ? kannadaNames[label] ?? label : label
}

function Icon({
  name,
  className = 'h-5 w-5',
}: {
  name: 'upload' | 'camera' | 'spark' | 'arrow' | 'refresh' | 'check'
  className?: string
}) {
  const paths = {
    upload: (
      <>
        <path d="M12 16V4m0 0 4 4m-4-4L8 8" />
        <path d="M4 16.5v2A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5v-2" />
      </>
    ),
    camera: (
      <>
        <path d="M4 7.5h3l1.5-2h7l1.5 2h3A1.5 1.5 0 0 1 21.5 9v9A1.5 1.5 0 0 1 20 19.5H4A1.5 1.5 0 0 1 2.5 18V9A1.5 1.5 0 0 1 4 7.5Z" />
        <circle cx="12" cy="13.5" r="3.5" />
      </>
    ),
    spark: (
      <>
        <path d="m12 2 1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2Z" />
        <path d="m19 15 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z" />
      </>
    ),
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    refresh: (
      <>
        <path d="M20 7v5h-5" />
        <path d="M20 12a8 8 0 1 0 2 5.3" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
  }
  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
    >
      {paths[name]}
    </svg>
  )
}

function App() {
  const [language, setLanguage] = useState<Language>('en')
  const [page, setPage] = useState<'identify' | 'metrics'>('identify')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [references, setReferences] = useState<Reference[]>([])
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)
  const cameraInput = useRef<HTMLInputElement>(null)
  const t = copy[language]

  useEffect(() => {
    let active = true
    Promise.all([
      fetch('/api/metrics').then(async (response) => {
        if (!response.ok) throw new Error('metrics')
        return (await response.json()) as Metrics
      }),
      fetch('/api/references').then(async (response) => {
        if (!response.ok) throw new Error('references')
        return (await response.json()) as Reference[]
      }),
    ])
      .then(([modelMetrics, modelReferences]) => {
        if (!active) return
        setMetrics(modelMetrics)
        setReferences(modelReferences)
      })
      .catch(() => {
        if (active) setError(t.apiUnavailable)
      })
    return () => {
      active = false
    }
  }, [t.apiUnavailable])

  useEffect(() => {
    if (previewUrl) return () => URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const topScores = useMemo(
    () => prediction?.class_scores.slice(0, 2) ?? [],
    [prediction],
  )

  function chooseFile(next: File | undefined) {
    if (!next) return
    setError('')
    setPrediction(null)
    const supported = ['image/jpeg', 'image/png', 'image/webp'].includes(
      next.type,
    )
    if (!supported) {
      setFile(null)
      setPreviewUrl(null)
      setError(
        language === 'kn'
          ? 'JPG, PNG ಅಥವಾ WEBP ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ.'
          : 'Choose a JPG, PNG, or WEBP image.',
      )
      return
    }
    if (next.size > 10 * 1024 * 1024) {
      setFile(null)
      setPreviewUrl(null)
      setError(
        language === 'kn'
          ? 'ಚಿತ್ರದ ಗಾತ್ರ 10 MB ಮೀರಬಾರದು.'
          : 'Image must be 10 MB or smaller.',
      )
      return
    }
    setFile(next)
    setPreviewUrl(URL.createObjectURL(next))
  }

  async function identify() {
    if (!file) return
    setBusy(true)
    setError('')
    setPrediction(null)
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': file.type },
        body: file,
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'Prediction failed.')
      }
      setPrediction(payload as Prediction)
    } catch (reason) {
      setError(
        reason instanceof Error && reason.message !== 'Failed to fetch'
          ? reason.message
          : t.apiUnavailable,
      )
    } finally {
      setBusy(false)
    }
  }

  function reset() {
    setFile(null)
    setPreviewUrl(null)
    setPrediction(null)
    setError('')
    if (fileInput.current) fileInput.current.value = ''
    if (cameraInput.current) cameraInput.current.value = ''
  }

  const acceptedConfidence =
    metrics?.uncertainty_thresholds.validation_accepted_accuracy ?? 0

  return (
    <div className="min-h-screen bg-ink text-parchment">
      <header className="border-b border-white/10">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 sm:px-8">
          <button
            className="flex items-center gap-3 text-left"
            onClick={() => setPage('identify')}
            aria-label={t.back}
          >
            <span className="brand-mark">ಯ</span>
            <span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.3em] text-saffron">
                Yakshagana
              </span>
              <span className="font-display text-lg leading-tight text-parchment">
                Vesha Identifier
              </span>
            </span>
          </button>
          <nav aria-label="Main navigation" className="flex items-center gap-2">
            <button
              className={`nav-link nav-desktop ${page === 'identify' ? 'nav-active' : ''}`}
              onClick={() => setPage('identify')}
            >
              {t.identify}
            </button>
            <button
              className={`nav-link nav-desktop ${page === 'metrics' ? 'nav-active' : ''}`}
              onClick={() => setPage('metrics')}
            >
              {t.performance}
            </button>
            <button
              className="language-toggle"
              onClick={() => setLanguage(language === 'en' ? 'kn' : 'en')}
              aria-label={language === 'en' ? 'ಕನ್ನಡಕ್ಕೆ ಬದಲಿಸಿ' : 'Switch to English'}
            >
              {t.bilingual}
            </button>
          </nav>
        </div>
        <div className="mx-auto max-w-7xl px-5 pb-3 sm:hidden">
          <button
            className={`nav-link w-full justify-center ${page === 'metrics' ? 'nav-active' : ''}`}
            onClick={() => setPage('metrics')}
          >
            {t.performance}
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 pb-20 pt-10 sm:px-8 sm:pt-14">
        {page === 'identify' ? (
          <>
            <section className="grid items-center gap-10 lg:grid-cols-[0.9fr_1.1fr] lg:gap-16">
              <div className="max-w-xl">
                <div className="eyebrow">
                  <span className="eyebrow-dot" />
                  {language === 'en'
                    ? 'A visual guide to Yakshagana'
                    : 'ಯಕ್ಷಗಾನದ ದೃಶ್ಯ ಪರಿಚಯ'}
                </div>
                <h1 className="mt-6 font-display text-5xl leading-[1.05] tracking-[-0.04em] text-parchment sm:text-6xl">
                  {t.tradition}
                </h1>
                <p className="mt-6 max-w-lg text-base leading-7 text-parchment/65">
                  {t.intro}
                </p>
                <div className="mt-7 flex flex-wrap items-center gap-3 text-xs text-parchment/55">
                  <span className="rounded-full border border-saffron/30 px-3 py-1.5 text-saffron">
                    {t.modelBadge}
                  </span>
                  <span>
                    {metrics
                      ? `${metrics.counts.test} ${t.test.toLowerCase()} images`
                      : t.loading}
                  </span>
                </div>
              </div>

              <div className="upload-card">
                <div className="flex items-start justify-between gap-5">
                  <div>
                    <span className="section-kicker">01 / {t.identify}</span>
                    <h2 className="mt-3 font-display text-3xl text-parchment">
                      {t.uploadTitle}
                    </h2>
                  </div>
                  <span className="icon-medallion">
                    <Icon name="spark" className="h-6 w-6" />
                  </span>
                </div>

                <div
                  className={`drop-zone mt-7 ${dragging ? 'drop-zone-active' : ''} ${previewUrl ? 'drop-zone-preview' : ''}`}
                  onDragOver={(event) => {
                    event.preventDefault()
                    setDragging(true)
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(event) => {
                    event.preventDefault()
                    setDragging(false)
                    chooseFile(event.dataTransfer.files[0])
                  }}
                >
                  {previewUrl ? (
                    <div className="preview-wrap">
                      <img
                        src={previewUrl}
                        alt={file?.name ?? t.yourPhoto}
                        className="preview-image"
                      />
                      <div className="absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-black/80 to-transparent p-4 pt-12">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-white">
                            {file?.name}
                          </p>
                          <p className="mt-1 text-xs text-white/65">
                            {file
                              ? `${(file.size / (1024 * 1024)).toFixed(2)} MB`
                              : ''}
                          </p>
                        </div>
                        <button
                          className="icon-button"
                          onClick={reset}
                          aria-label={t.reset}
                        >
                          <Icon name="refresh" />
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex min-h-[245px] flex-col items-center justify-center px-5 text-center">
                      <span className="upload-icon">
                        <Icon name="upload" className="h-7 w-7" />
                      </span>
                      <p className="mt-5 text-sm font-medium text-parchment">
                        {t.uploadHint}
                      </p>
                      <p className="mt-2 text-xs text-parchment/45">
                        {t.fileTypes}
                      </p>
                    </div>
                  )}
                </div>

                <input
                  ref={fileInput}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="sr-only"
                  onChange={(event) => chooseFile(event.target.files?.[0])}
                  aria-label={t.choose}
                />
                <input
                  ref={cameraInput}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  capture="environment"
                  className="sr-only"
                  onChange={(event) => chooseFile(event.target.files?.[0])}
                  aria-label={t.camera}
                />
                {!previewUrl && (
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <button
                      className="button-secondary"
                      onClick={() => fileInput.current?.click()}
                    >
                      {t.choose}
                    </button>
                    <button
                      className="button-secondary"
                      onClick={() => cameraInput.current?.click()}
                    >
                      <Icon name="camera" className="h-4 w-4" />
                      {t.camera}
                    </button>
                  </div>
                )}
                <p className="mt-4 text-center text-xs text-parchment/45">
                  {t.crop}
                </p>
                {error && (
                  <p className="mt-4 rounded-lg border border-red-400/30 bg-red-950/40 px-4 py-3 text-sm text-red-200" role="alert">
                    {error}
                  </p>
                )}
                {previewUrl && (
                  <button
                    className="button-primary mt-5 w-full"
                    disabled={busy}
                    onClick={identify}
                  >
                    {busy ? (
                      <>
                        <span className="spinner" />
                        {t.analysing}
                      </>
                    ) : (
                      <>
                        <Icon name="spark" className="h-5 w-5" />
                        {t.identifyButton}
                        <Icon name="arrow" className="ml-auto h-5 w-5" />
                      </>
                    )}
                  </button>
                )}
              </div>
            </section>

            <div className="mt-9 flex gap-3 rounded-xl border border-saffron/20 bg-saffron/[0.06] px-4 py-4 text-sm leading-6 text-parchment/70 sm:px-5">
              <span className="mt-0.5 text-saffron" aria-hidden="true">i</span>
              <p>{t.validationNotice}</p>
            </div>

            {prediction && previewUrl && (
              <section className="mt-14" aria-live="polite">
                <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
                  <div>
                    <span className="section-kicker">02 / {t.result}</span>
                    <h2 className="mt-2 font-display text-4xl text-parchment">
                      {t.comparison}
                    </h2>
                  </div>
                  <button className="button-quiet" onClick={reset}>
                    <Icon name="refresh" className="h-4 w-4" />
                    {t.reset}
                  </button>
                </div>

                <div className={`result-banner ${prediction.status}`}>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-saffron">
                      {t.result}
                    </p>
                    <h3 className="mt-2 font-display text-3xl text-parchment sm:text-4xl">
                      {prediction.status === 'uncertain'
                        ? t.uncertain
                        : prediction.status === 'not_yakshagana'
                          ? t.notYak
                          : `${t.yak}: ${nameFor(prediction.predicted_class ?? '', language)}`}
                    </h3>
                    {prediction.confidence === null ? (
                      <p className="mt-3 max-w-xl text-sm leading-6 text-parchment/65">
                        {t.datasetMismatch}
                      </p>
                    ) : (
                      <p className="mt-3 text-sm text-parchment/65">
                        {t.confidence}: {(prediction.confidence * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                  {prediction.confidence !== null && (
                    <div className="confidence-orbit" aria-hidden="true">
                      <span>{Math.round(prediction.confidence * 100)}</span>
                      <small>%</small>
                    </div>
                  )}
                </div>
                {prediction.status !== 'not_yakshagana' && (
                  <p className="mt-3 text-xs text-parchment/45">
                    {t.confidenceNote}
                  </p>
                )}

                <div
                  className={`mt-8 grid gap-8 ${prediction.status === 'not_yakshagana' ? '' : 'xl:grid-cols-[1.15fr_0.85fr]'}`}
                >
                  <div className="panel">
                    <div className="mb-5 flex items-center justify-between">
                      <div>
                        <span className="section-kicker">03 / {t.comparison}</span>
                        <h3 className="mt-2 font-display text-2xl">
                          {t.topChoices}
                        </h3>
                      </div>
                      <span className="text-xs text-parchment/45">
                        {t.reference}
                      </span>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-3">
                      <figure className="compare-figure">
                        <img src={previewUrl} alt={t.yourPhoto} />
                        <figcaption>
                          <span>{t.yourPhoto}</span>
                          <span>{file?.name}</span>
                        </figcaption>
                      </figure>
                      {prediction.status === 'not_yakshagana' ? (
                        <div className="reference-placeholder px-5 text-center text-sm leading-6 text-parchment/65">
                          {t.datasetMismatch}
                        </div>
                      ) : topScores.map((score) => {
                        const referenceIndex = references.findIndex(
                          (reference) => reference.label === score.class,
                        )
                        return (
                          <figure className="compare-figure" key={score.class}>
                            {referenceIndex >= 0 ? (
                              <img
                                src={references[referenceIndex].url}
                                alt={`${t.reference}: ${nameFor(score.class, language)}`}
                              />
                            ) : (
                              <div className="reference-placeholder">
                                {nameFor(score.class, language)}
                              </div>
                            )}
                            <figcaption>
                              <span>{nameFor(score.class, language)}</span>
                              <span>{(score.score * 100).toFixed(1)}%</span>
                            </figcaption>
                          </figure>
                        )
                      })}
                    </div>
                  </div>

                  {prediction.status !== 'not_yakshagana' && (
                    <div className="panel">
                      <div className="mb-5">
                        <span className="section-kicker">04 / {t.classScores}</span>
                        <h3 className="mt-2 font-display text-2xl">{t.allScores}</h3>
                      </div>
                      <div className="space-y-4">
                        {prediction.class_scores.map((score, index) => (
                          <div key={score.class}>
                            <div className="mb-1.5 flex justify-between gap-3 text-xs">
                              <span className="truncate text-parchment/80">
                                <span className="mr-2 text-saffron/70">
                                  {String(index + 1).padStart(2, '0')}
                                </span>
                                {nameFor(score.class, language)}
                              </span>
                              <span className="tabular-nums text-parchment/60">
                                {(score.score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div
                              className="score-track"
                              role="progressbar"
                              aria-label={`${score.class} ${t.confidence}`}
                              aria-valuenow={Math.round(score.score * 100)}
                              aria-valuemin={0}
                              aria-valuemax={100}
                            >
                              <span
                                className={index === 0 ? 'score-fill score-fill-top' : 'score-fill'}
                                style={{ width: `${score.score * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                {prediction.status !== 'not_yakshagana' && (
                  <p className="mt-4 text-xs text-parchment/40">{t.confidenceNote}</p>
                )}
              </section>
            )}
          </>
        ) : (
          <section>
            <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
              <div>
                <span className="section-kicker">01 / {t.performance}</span>
                <h1 className="mt-3 font-display text-5xl text-parchment">
                  {t.performance}
                </h1>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-parchment/60">
                  {metrics?.model} · {metrics?.counts.train} {t.train.toLowerCase()} / {metrics?.counts.validation} {t.validation.toLowerCase()} / {metrics?.counts.test} {t.test.toLowerCase()}
                </p>
              </div>
              <button className="button-quiet" onClick={() => setPage('identify')}>
                {t.back}
              </button>
            </div>

            {error && !metrics ? (
              <div className="panel text-red-200" role="alert">
                <h2 className="font-display text-2xl">{t.errorTitle}</h2>
                <p className="mt-3 text-sm">{error}</p>
              </div>
            ) : !metrics ? (
              <div className="panel">{t.loading}</div>
            ) : (
              <>
                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  <MetricCard
                    title={t.metricAccuracy}
                    value={`${(metrics.accuracy * 100).toFixed(1)}%`}
                    note={`${metrics.counts.test} ${t.test.toLowerCase()} images`}
                    highlight
                  />
                  <MetricCard
                    title={t.validationAccuracy}
                    value={`${(acceptedConfidence * 100).toFixed(1)}%`}
                    note={`${metrics.uncertainty_thresholds.target_accepted_accuracy * 100}% validation target`}
                  />
                  <MetricCard
                    title={t.uncertainTest}
                    value={`${metrics.uncertain_count} / ${metrics.counts.test}`}
                    note={`${(metrics.uncertain_rate * 100).toFixed(1)}% of held-out test images`}
                  />
                  <MetricCard
                    title={t.yakImagesRejected}
                    value={`${metrics.novelty_screen.test_positive_rejection_count} / ${metrics.counts.test}`}
                    note={`${(metrics.novelty_screen.test_positive_rejection_rate * 100).toFixed(1)}% of held-out Yakshagana images`}
                  />
                </div>
                <p className="mt-4 text-xs leading-5 text-parchment/45">
                  {t.confidenceNote}
                </p>

                <div className="mt-8 grid gap-8 xl:grid-cols-[1.1fr_0.9fr]">
                  <div className="panel overflow-hidden">
                    <div className="mb-5">
                      <span className="section-kicker">02 / {t.classPerformance}</span>
                      <h2 className="mt-2 font-display text-2xl">{t.classPerformance}</h2>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="metrics-table">
                        <thead>
                          <tr>
                            <th>{t.class}</th>
                            <th>{t.images}</th>
                            <th>{t.precision}</th>
                            <th>{t.recall}</th>
                            <th>{t.f1}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {metrics.classes.map((item) => (
                            <tr key={item.label}>
                              <td className="font-medium text-parchment">
                                {nameFor(item.label, language)}
                              </td>
                              <td>{item.image_count}</td>
                              <td>{(item.precision * 100).toFixed(1)}%</td>
                              <td>{(item.recall * 100).toFixed(1)}%</td>
                              <td>{(item.f1 * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="mt-4 text-xs text-parchment/45">
                      {t.images}: full usable dataset count · class scores use held-out test images.
                    </p>
                  </div>

                  <div className="panel">
                    <div className="mb-5">
                      <span className="section-kicker">03 / {t.confusion}</span>
                      <h2 className="mt-2 font-display text-2xl">{t.confusion}</h2>
                    </div>
                    <div className="overflow-x-auto">
                      <ConfusionMatrix
                        metrics={metrics}
                        language={language}
                        actual={t.actual}
                        predicted={t.predicted}
                      />
                    </div>
                  </div>
                </div>

                <div className="mt-8 rounded-xl border border-amber-400/25 bg-amber-950/25 p-5 sm:p-6">
                  <h2 className="font-display text-2xl text-saffron">
                    {t.nonYakTitle}
                  </h2>
                  <p className="mt-3 text-sm leading-6 text-parchment/65">
                    {t.nonYakUnavailable}
                  </p>
                  <p className="mt-3 text-xs leading-5 text-parchment/45">
                    Validation accepted {(metrics.novelty_screen.validation_positive_acceptance_rate * 100).toFixed(1)}% of positive validation images. Test-set Model accuracy includes Yakshagana images rejected by this screen. The six-class model alone scored {(metrics.classifier_accuracy * 100).toFixed(1)}%.
                  </p>
                </div>
              </>
            )}
          </section>
        )}
      </main>

      <footer className="border-t border-white/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-5 py-6 text-xs text-parchment/40 sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <span>{t.footer}</span>
          <span>{t.footerAccuracy}</span>
        </div>
      </footer>
    </div>
  )
}

function MetricCard({
  title,
  value,
  note,
  highlight = false,
}: {
  title: string
  value: string
  note: string
  highlight?: boolean
}) {
  return (
    <article className={`metric-card ${highlight ? 'metric-card-highlight' : ''}`}>
      <p className="text-xs uppercase tracking-[0.15em] text-parchment/50">{title}</p>
      <p className="mt-4 font-display text-4xl text-parchment">{value}</p>
      <p className="mt-2 text-xs leading-5 text-parchment/45">{note}</p>
    </article>
  )
}

function ConfusionMatrix({
  metrics,
  language,
  actual,
  predicted,
}: {
  metrics: Metrics
  language: Language
  actual: string
  predicted: string
}) {
  const values = metrics.confusion_matrix.values
  const maximum = Math.max(1, ...values.flat())
  return (
    <table className="confusion-table">
      <caption className="sr-only">
        {actual} by {predicted}
      </caption>
      <thead>
        <tr>
          <th className="axis-label">{actual} ↓<br />{predicted} →</th>
          {metrics.confusion_matrix.predicted_labels.map((label) => (
            <th key={label} title={nameFor(label, language)}>
              {nameFor(label, language)}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {values.map((row, rowIndex) => (
          <tr key={metrics.confusion_matrix.actual_labels[rowIndex]}>
            <th title={nameFor(metrics.confusion_matrix.actual_labels[rowIndex], language)}>
              {nameFor(metrics.confusion_matrix.actual_labels[rowIndex], language)}
            </th>
            {row.map((count, columnIndex) => {
              const isDiagonal = rowIndex === columnIndex
              return (
                <td
                  key={`${rowIndex}-${columnIndex}`}
                  className={isDiagonal ? 'matrix-correct' : count ? 'matrix-error' : ''}
                  style={{
                    backgroundColor: count
                      ? `rgba(216, 173, 91, ${0.08 + (count / maximum) * 0.42})`
                      : undefined,
                  }}
                >
                  {count}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default App
