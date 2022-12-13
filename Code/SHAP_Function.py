import shap


def shap_explanation(train_df, trained_model, tokenizer):
    df = train_df['text'][0:1]
    explainer = shap.Explainer(trained_model, tokenizer)
    shap_values = explainer(df)
    # shap.plots.text(shap_values)

    # html_string = (shap.plots.text(shap_values)).data
    #
    # with open("explainability.html", "w") as file:
    #     file.write(shap.plots.text(shap_values))

    file = open('explainability.html', 'w')
    y = shap.plots.text(shap_values, display=False)
    # result = (re.sub(r"<" + s + ">(.*?)</" + s + ">", "<" + p + ">(.*?)</" + p + ">, y))
    clean_HTML = y.replace('<s>', '<p>').replace('</s>', '</p>')
    file.write(clean_HTML)
    file.close
