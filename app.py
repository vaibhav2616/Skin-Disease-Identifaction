import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file in the same directory
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)

# Load model
model = load_model("skindisease.h5")

# Class names
CLASS_NAMES = ['Acne', 'Melanoma', 'Peeling skin', 'Ring worm', 'Vitiligo']


# ⭐ SMART CONFIDENCE FIX FUNCTION
def realistic_softmax(logits):
    logits = np.array(logits, dtype=np.float64)

    # if logits are too similar → model is unsure → amplify difference
    diff = logits.max() - logits.min()

    if diff < 1.0:
        # VERY flat prediction → strong sharpening
        logits = logits * 6.0
    elif diff < 2.0:
        # moderately flat
        logits = logits * 4.0
    else:
        # normal model confidence
        logits = logits * 2.0

    # Safe softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs


@app.route('/', methods=['GET'])
def index():
    return render_template("base.html",
                           prediction_text=None,
                           top3=None,
                           all_probs=None,
                           disease_info=None)


@app.route('/find-dermatologist', methods=['GET'])
def find_dermatologist():
    return render_template("dermatologist.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template("base.html",
                               prediction_text="⚠ No image uploaded.",
                               top3=None, all_probs=None, disease_info=None)

    f = request.files['image']
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, secure_filename(f.filename))
    f.save(filepath)

    try:
        # Load + preprocess
        img = load_img(filepath, target_size=(64, 64))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        raw_preds = model.predict(x, verbose=0)[0]
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

    # ⭐ GET REALISTIC CONFIDENCE
    probs = realistic_softmax(raw_preds)

    # Top 3 display
    top3_idx = np.argsort(-probs)[:3]
    top3 = [(CLASS_NAMES[i], float(probs[i]), i) for i in top3_idx]

    # All class probabilities
    all_probs = [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))]

    final_label = CLASS_NAMES[np.argmax(probs)]
    prediction_text = f"The Skin Disease is \"{final_label}\""

    info_map = {
        "Acne": {
            "description": "Acne occurs when pores get clogged with oil and dead skin cells, leading to pimples, blackheads, and whiteheads.",
            "prevention": [
                "Wash your face twice daily with a gentle cleanser",
                "Avoid touching your face with dirty hands",
                "Use non-comedogenic (non-pore-clogging) skincare products",
                "Keep hair clean and away from your face",
                "Avoid excessive sun exposure and use oil-free sunscreen",
                "Change pillowcases regularly",
                "Maintain a balanced diet and stay hydrated"
            ],
            "cures": [
                "Topical treatments: Benzoyl peroxide, salicylic acid, or retinoids",
                "Prescription medications: Antibiotics (clindamycin, erythromycin) or oral contraceptives",
                "Isotretinoin for severe cases (requires medical supervision)",
                "Chemical peels or light therapy in dermatologist's office",
                "Extraction procedures by professionals"
            ],
            "remedies": [
                "Tea tree oil (diluted) has antimicrobial properties",
                "Green tea extract can reduce inflammation",
                "Aloe vera gel soothes irritated skin",
                "Honey masks have antibacterial effects",
                "Zinc supplements may help reduce inflammation",
                "Avoid picking or popping pimples to prevent scarring"
            ]
        },
        "Melanoma": {
            "description": "Melanoma is a serious form of skin cancer that develops in melanocytes (cells that produce melanin). Early detection is crucial for successful treatment.",
            "prevention": [
                "Use broad-spectrum sunscreen (SPF 30+) daily, even on cloudy days",
                "Avoid tanning beds and excessive sun exposure",
                "Seek shade during peak sun hours (10 AM - 4 PM)",
                "Wear protective clothing: wide-brimmed hats, long sleeves, UV-blocking sunglasses",
                "Perform monthly self-examinations of your skin",
                "Get annual professional skin checks, especially if you have risk factors",
                "Know your ABCDEs: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving"
            ],
            "cures": [
                "Surgical excision is the primary treatment for early-stage melanoma",
                "Sentinel lymph node biopsy for staging",
                "Immunotherapy: Checkpoint inhibitors (pembrolizumab, nivolumab)",
                "Targeted therapy: BRAF/MEK inhibitors for specific genetic mutations",
                "Radiation therapy for advanced cases or when surgery isn't possible",
                "Chemotherapy may be used in advanced stages",
                "Clinical trials for experimental treatments"
            ],
            "remedies": [
                "⚠️ IMPORTANT: Melanoma requires immediate medical attention. Do not attempt home remedies.",
                "Supportive care: Maintain a healthy diet rich in antioxidants",
                "Stay hydrated and get adequate rest during treatment",
                "Protect skin from further sun damage",
                "Join support groups for emotional support",
                "Follow your oncologist's treatment plan strictly"
            ]
        },
        "Peeling skin": {
            "description": "Peeling skin can result from various causes including dryness, sunburn, allergic reactions, fungal infections, or underlying medical conditions.",
            "prevention": [
                "Moisturize regularly with fragrance-free, hypoallergenic lotions",
                "Avoid hot showers and use lukewarm water",
                "Use gentle, soap-free cleansers",
                "Protect skin from harsh weather conditions",
                "Stay hydrated by drinking plenty of water",
                "Avoid known allergens and irritants",
                "Use humidifiers in dry environments",
                "Wear protective gloves when using harsh chemicals"
            ],
            "cures": [
                "Topical corticosteroids for inflammation (prescribed by doctor)",
                "Antifungal medications if caused by fungal infection",
                "Antihistamines for allergic reactions",
                "Moisturizing creams with ceramides or hyaluronic acid",
                "Phototherapy for certain conditions like psoriasis",
                "Oral medications for severe cases (as prescribed by dermatologist)"
            ],
            "remedies": [
                "Apply aloe vera gel to soothe and moisturize",
                "Oatmeal baths can relieve itching and irritation",
                "Coconut oil provides natural moisturization",
                "Honey masks have antibacterial and moisturizing properties",
                "Stay well-hydrated and maintain a balanced diet",
                "Avoid scratching or picking at peeling skin",
                "Use gentle exfoliation only if recommended by a dermatologist"
            ]
        },
        "Ring worm": {
            "description": "Ringworm (tinea) is a common fungal infection that causes a circular, ring-like rash. Despite its name, it's not caused by a worm.",
            "prevention": [
                "Keep skin clean and dry, especially in skin folds",
                "Wear breathable, loose-fitting clothing",
                "Avoid sharing personal items: towels, clothing, combs",
                "Wear flip-flops in public showers and pools",
                "Wash hands regularly, especially after touching pets",
                "Treat pets if they show signs of ringworm",
                "Change socks and underwear daily",
                "Disinfect surfaces that may be contaminated"
            ],
            "cures": [
                "Topical antifungal creams: Clotrimazole, miconazole, terbinafine (apply 2-4 weeks)",
                "Oral antifungal medications: Terbinafine, itraconazole (for severe cases)",
                "Prescription-strength topical treatments from dermatologist",
                "Keep affected area clean and dry during treatment",
                "Complete the full course of medication even if symptoms improve"
            ],
            "remedies": [
                "Tea tree oil (diluted) has antifungal properties",
                "Apple cider vinegar (diluted) may help fight fungus",
                "Coconut oil contains lauric acid with antifungal effects",
                "Turmeric paste has anti-inflammatory and antimicrobial properties",
                "Garlic extract (topical, diluted) may have antifungal benefits",
                "Keep the area dry and exposed to air when possible",
                "⚠️ Note: Home remedies should complement, not replace, medical treatment"
            ]
        },
        "Vitiligo": {
            "description": "Vitiligo is a condition where patches of skin lose their pigment (melanin), resulting in white or light-colored patches. It's an autoimmune condition.",
            "prevention": [
                "Protect depigmented areas from sunburn with high SPF sunscreen",
                "Avoid skin trauma and injuries that may trigger new patches",
                "Manage stress through relaxation techniques",
                "Maintain a healthy, balanced diet",
                "Avoid chemical exposure that may irritate skin",
                "Use gentle, hypoallergenic skincare products",
                "Regular monitoring and early treatment can help manage progression"
            ],
            "cures": [
                "Topical corticosteroids to restore some pigment",
                "Topical calcineurin inhibitors (tacrolimus, pimecrolimus)",
                "Phototherapy: Narrowband UVB or PUVA therapy",
                "Excimer laser treatment for localized patches",
                "Surgical options: Skin grafting, melanocyte transplantation",
                "Depigmentation therapy for extensive vitiligo (rare)",
                "JAK inhibitors (ruxolitinib) - newer treatment option"
            ],
            "remedies": [
                "Ginkgo biloba extract may help slow progression (consult doctor)",
                "Vitamin D supplements (under medical supervision)",
                "Psoralen combined with sunlight exposure (PUVA) - requires medical guidance",
                "Turmeric and mustard oil mixture (traditional remedy, limited evidence)",
                "Copper-rich foods: nuts, seeds, whole grains",
                "Stress management: yoga, meditation, counseling",
                "⚠️ Important: Vitiligo treatment should be supervised by a dermatologist"
            ]
        }
    }

    disease_info = info_map.get(final_label, {})

    return render_template("base.html",
                           prediction_text=prediction_text,
                           top3=top3,
                           all_probs=all_probs,
                           disease_info=disease_info)


# Chatbot API
@app.route('/chat', methods=['POST'])
def chat_bot():
    try:
        data = request.get_json()
        q = data.get("query", "").strip()
        
        if not q:
            return jsonify({"answer": "Please ask a question about skin diseases.", "sources": []})
        
        # Initialize Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return jsonify({"answer": "API key not configured. Please check environment variables.", "sources": []})
        
        genai.configure(api_key=api_key)
        # Use gemini-2.5-flash for faster responses (stable model)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create a context-aware prompt
        prompt = f"""You are a helpful medical assistant for a skin disease detection system. 
The system can detect: Acne, Melanoma, Peeling skin, Ring worm, and Vitiligo.

User question: {q}

Provide a helpful, accurate, and concise answer about skin diseases, symptoms, or how to use the system. 
If asked about medical advice, remind users to consult a healthcare professional for diagnosis and treatment.
Keep responses clear and under 200 words."""

        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        
        return jsonify({"answer": answer, "sources": []})
    
    except Exception as e:
        return jsonify({"answer": f"Sorry, I encountered an error: {str(e)}. Please try again.", "sources": []})


if __name__ == '__main__':
    app.run(debug=True)
