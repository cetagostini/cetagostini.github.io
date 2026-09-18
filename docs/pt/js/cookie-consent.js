// Cookie Consent Handler
document.addEventListener('DOMContentLoaded', function() {
  // Check if consent was already given
  const consentGiven = localStorage.getItem('cookieConsent');
  
  // If not already decided, show the popup
  if (consentGiven === null) {
    showCookieConsent();
  } else if (consentGiven === 'accepted') {
    // If consent was previously given, load analytics
    loadGoogleAnalytics();
  }
});

// Function to show the cookie consent popup
function showCookieConsent() {
  // Create cookie consent container
  const consentContainer = document.createElement('div');
  consentContainer.className = 'cookie-consent';
  
  // Create content
  var _t = window.siteI18n ? window.siteI18n.t.bind(window.siteI18n) : function (k) { return k; };
  consentContainer.innerHTML = `
    <div class="cookie-content">
      <h3>${_t("cookie.title")}</h3>
      <p>${_t("cookie.body")}</p>
      <div class="cookie-buttons">
        <button id="cookie-accept" class="btn btn-primary">${_t("cookie.accept")}</button>
        <button id="cookie-decline" class="btn btn-outline-primary">${_t("cookie.decline")}</button>
      </div>
    </div>
  `;
  
  // Add to body
  document.body.appendChild(consentContainer);
  
  // Add event listeners for buttons
  document.getElementById('cookie-accept').addEventListener('click', function() {
    localStorage.setItem('cookieConsent', 'accepted');
    consentContainer.classList.add('cookie-hidden');
    
    // Remove after animation completes
    setTimeout(function() {
      document.body.removeChild(consentContainer);
    }, 500);
    
    // Load Google Analytics
    loadGoogleAnalytics();
  });
  
  document.getElementById('cookie-decline').addEventListener('click', function() {
    localStorage.setItem('cookieConsent', 'declined');
    consentContainer.classList.add('cookie-hidden');
    
    // Remove after animation completes
    setTimeout(function() {
      document.body.removeChild(consentContainer);
    }, 500);
  });
}

// Function to load Google Analytics
function loadGoogleAnalytics() {
  // Google Tag Manager implementation
  (function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
  new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
  j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
  'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
  })(window,document,'script','dataLayer','GTM-NL6GRRJ3');
  
  // Add noscript iframe
  const noscript = document.createElement('noscript');
  const iframe = document.createElement('iframe');
  iframe.src = "https://www.googletagmanager.com/ns.html?id=GTM-NL6GRRJ3";
  iframe.height = "0";
  iframe.width = "0";
  iframe.style.display = "none";
  iframe.style.visibility = "hidden";
  noscript.appendChild(iframe);
  document.body.appendChild(noscript);
} 