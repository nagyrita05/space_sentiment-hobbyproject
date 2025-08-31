
# Policy/Finance altéma médiaképének értékelése

## 🎯 Cél
Az elemzés célja, hogy összevessük a cikkcímek hangulatának manuális (emberi értelmezésű) és automatikus (kulcsszavas) besorolását, majd országonkénti bontásban is vizualizáljuk az eredményeket.

## 🧪 Módszertan

### Manuális értékelés
- 30 véletlenszerűen kiválasztott cikkcímet vizsgáltunk.
- Minden címet emberi olvasatban értékeltünk három kategóriában:
  - 🟢 pozitív
  - ⚪ semleges
  - 🔴 negatív
- A végső arányok:
  - Pozitív: 40%
  - Semleges: 40%
  - Negatív: 20%

### Automatikus értékelés (kulcsszavas módszerrel)
- Az összes (`n = 689`) cikkcímet vizsgáltuk.
- Kulcsszavak alapján történt a címek besorolása.
- Eredmények:
  - Pozitív: 17,7%
  - Semleges: 78,7%
  - Negatív: 3,6%

### Országonkénti térképes vizualizáció
- Az országonkénti `positive`, `neutral`, `negative` arányokból számított sentiment score:

  \\[
  	ext{sentiment\_score} = \frac{positive - negative}{total}
  \\]

- A score -1 és +1 között mozog (piros = negatív, zöld = pozitív).

## 🗺️ Vizualizáció

![policy_finance_sentiment_map](policy_finance_sentiment_map.png)

> A térkép azt mutatja, hogy egyes országokban inkább pozitív, máshol inkább negatív hangvételűek a cikkcímek a policy/finance témában.

## 🧠 Kulcskülönbségek manuális vs automatikus értékelés között

| Jellemző | Manuális értékelés (ChatGPT) | Automatikus (kulcsszavas) |
|---------|-------------------------------|----------------------------|
| Kontextusérzékenység | ✅ Igen | ⚠️ Korlátozott |
| Árnyalatok érzékelése | ✅ Igen (pl. társadalmi szempontból) | ❌ Nem |
| Skálázhatóság | ❌ Lassabb | ✅ Teljes fájl |
| Jellemző torzítás | Kis elemszámnál kiugró értékek | Semleges túlsúly |

## 🧾 Következtetés

- A manuális értékelés érzékenyebb a társadalmi és érzelmi árnyalatokra (pl. nők űrutazása, szimbolikus jelentések).
- Az automatikus gyorsabb és skálázhatóbb, de leegyszerűsíti a jelentéseket.
- Érdemes őket kombinálni a pontosabb médiakép megértéséhez.

