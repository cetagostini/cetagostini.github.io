// Source of files/cv_carlos.pdf. Build from the repo root, with the Inter and
// Manrope font files (Google Fonts) in <font-dir>:
//   typst compile --root . --pdf-standard ua-1 --font-path <font-dir> cv/carlos-trujillo.typ files/cv_carlos.pdf
// Palette: cetagostini.github.io/styles.css.
#let paper = rgb("#FDF6ED")
#let sage = rgb("#778873")
#let green = rgb("#4F6B4A")
#let ink = rgb("#2B2A26")
#let muted = rgb("#6B665C")
#let brown = rgb("#6B5A48")
#let rule = rgb("#DCCFC0")
#let site = "https://cetagostini.github.io"
#let labs = "https://www.pymc-labs.com"
#let email = "carlos.trujillo.agostini@gmail.com"

// One grid for the whole page: labels and portrait on the left, content on the right.
#let label-col = 34mm
#let gutter = 8mm

#set document(title: "Carlos Trujillo | CV", author: "Carlos Trujillo", keywords: ("Bayesian inference", "Causal modeling", "Time series"))
#set page(paper: "a4", margin: (x: 18mm, y: 17mm), fill: paper)
#set text(font: "Inter", size: 9.4pt, fill: ink, lang: "en")
#set par(leading: 0.62em, spacing: 0.62em)
#show link: set text(fill: green)
#show heading.where(level: 1): set text(font: "Manrope", size: 10.5pt, weight: "semibold", fill: brown)
#show heading.where(level: 1): set block(above: 0pt, below: 0pt)
#show heading.where(level: 2): set text(font: "Manrope", size: 10.5pt, weight: "semibold", fill: ink)
#show heading.where(level: 2): set block(above: 0pt, below: 0pt)

#let meta(body) = text(size: 8.6pt, fill: muted, body)
#let dot = text(fill: sage)[#h(5pt)·#h(5pt)]

#let section(label, body) = block(above: 0pt, below: 0pt)[
  #line(length: 100%, stroke: 0.5pt + rule)
  #v(11pt)
  #grid(columns: (label-col, 1fr), column-gutter: gutter,
    heading(level: 1, label), body,
  )
  #v(15pt)
]

#let entry(org, dates, role, body: none) = block(breakable: false, above: 0pt, below: 14pt)[
  #grid(columns: (1fr, auto), column-gutter: 6pt, align: bottom,
    heading(level: 2, org), meta(dates),
  )
  #v(3pt)
  #text(size: 9pt, weight: "medium", fill: green, role)
  #if body != none [
    #v(3pt)
    #body
  ]
]

#let row(left, right) = grid(columns: (1fr, auto), column-gutter: 8pt, left, meta(right))

// Header
#grid(columns: (label-col, 1fr), column-gutter: gutter, align: horizon,
  box(stroke: 0.7pt + sage, radius: 50%, inset: 1.2mm)[
    #box(width: 29mm, height: 29mm, radius: 50%, clip: true,
      image("/images/profile.jpg", width: 100%, height: 100%, fit: "cover", alt: "Portrait of Carlos Trujillo"),
    )
  ],
  [
    #title(text(font: "Manrope", size: 27pt, weight: "semibold", fill: ink)[Carlos Trujillo])
    #v(4pt)
    #text(size: 10.5pt, weight: "semibold", fill: green)[Probabilistic and causal modeling thinker]
    #v(7pt)
    #block(width: 120mm)[
      Leading research that connects Bayesian inference, causal modeling, and time series with neural networks to solve business problems.
    ]
    #v(7pt)
    #text(size: 8.8pt)[
      #link("mailto:" + email, email) #dot
      #link(site)[cetagostini.github.io] #dot
      #link("https://github.com/cetagostini")[GitHub] #dot
      #link("https://www.linkedin.com/in/cetagostini/")[LinkedIn]
    ]
  ],
)

#v(10mm)

#section([Experience], [
  #entry([PyMC Labs], [2024 – present], [Principal Data Scientist #dot Tallinn, Estonia], body: [
    Lead research on statistical foundation models with prior-data fitted networks (PFNs) and amortized Bayesian inference. Advise clients on causal and time-series models for marketing measurement and budget decisions. Open-source contributor to #link("https://github.com/pymc-devs/pymc")[PyMC], #link("https://github.com/pymc-labs/pymc-marketing")[PyMC-Marketing], #link("https://github.com/pymc-devs/pytensor")[PyTensor], and #link("https://github.com/pymc-labs/CausalPy")[CausalPy].
  ])
  #entry([Wise], [2024 – 2026], [Lead Data Scientist #dot Senior Marketing Scientist #dot Tallinn, Estonia], body: [
    Led marketing measurement strategy, owning media mix modeling and incrementality testing in PyMC. Worked with marketing leadership to turn model results into budget decisions.
  ])
  #entry([Bolt], [2022 – 2024], [Marketing Ads, Functional Analyst #dot Tallinn, Estonia], body: [
    Built media mix, causal impact, and budget allocation models for acquisition marketing across Europe and South Africa.
  ])
  #entry([Omnicom Media Group], [2020 – 2022], [Head of Data Analytics #dot Data Analytics Manager #dot Santiago, Chile], body: [
    Led the Marketing Science LATAM data team and built regional analytics services on AWS and Google Cloud.
  ])
  #entry([Earlier], [2018 – 2020], [Data Analyst at Raya #dot Performance Manager at OYM Agencia])
])

#section([Expertise], [
  #grid(columns: (17mm, 1fr), column-gutter: 4mm, row-gutter: 7pt,
    meta[Methods], [Bayesian inference, amortized inference, causal inference, time series, media mix modeling, Bayesian optimization],
    meta[Tools], [Python, PyMC, PyMC-Marketing, PyTensor, CausalPy, R, SQL, Databricks, Git],
  )
])

#section([Selected work], [
  #set block(spacing: 7pt)
  #row(link(site + "/articles/cross_city_media_spillovers/cross_city_media_spillovers.html")[Cross-city media spillovers], [Article, 2026])
  #row(link(site + "/articles/placebo_bayesian_quasi_experiments/placebo_bayesian_quasi_experiments.html")[Auditing quasi-experiments with Bayesian placebo tests], [Article, 2026])
  #row(link(labs + "/blog-posts/probabilistic-arrival-time-forecasting-deadline-decisions")[Probabilistic arrival-time forecasting], [PyMC Labs, 2026])
  #row(link("https://www.youtube.com/watch?v=wEtOkvV43es")[Evaluating LLM agents for causal data science], [Causal Summit, 2026])
  #row(link(labs + "/courses/bayesian-marketing-analytics")[Bayesian Marketing Analytics], [PyMC Labs course, co-instructor])
])

#section([Education], [
  #set block(spacing: 7pt)
  #row([Bachelor’s in Market Studies #dot Universidad José Antonio Páez], [2015 – 2019])
  #row([Data Science training #dot Acámica], [2020])
])

#block(above: 0pt)[
  #line(length: 100%, stroke: 0.5pt + rule)
  #v(10pt)
  #grid(columns: (label-col, 1fr), column-gutter: gutter,
    heading(level: 1)[Languages],
    [Spanish, C2 #dot English, C1 #dot German, A2 #dot Estonian, A2],
  )
]
