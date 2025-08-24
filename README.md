# Trading Tool Pro

Outil de trading professionnel pour la gestion et l'analyse de données financières.

## Fonctionnalités

- 📊 Récupération et mise à jour automatique des données depuis First Rate Data
- 💾 Stockage des données en format Parquet avec partitionnement par année
- 🔄 Remplissage automatique des gaps de trading
- 📈 Calcul de moyennes de prix par période
- 📅 Gestion des calendriers de trading par marché
- 🌍 Support multi-assets : futures, stocks, ETFs, forex, cryptos
- ⏰ Gestion des heures de trading (RTH, ETH)

## Installation

1. **Cloner le projet**
```bash
git clone <votre-repo>
cd tradingtool2
```

2. **Activer l'environnement virtuel**
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Installer les dépendances**
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Lancement de l'Application

### 🪟 Windows
```bash
# Option 1: Script batch
launch.bat

# Option 2: Script PowerShell
powershell -ExecutionPolicy Bypass -File launch.ps1

# Option 3: Manuel
.venv\Scripts\streamlit.exe run app.py
```

### 🐧 Linux/Mac
```bash
# Option 1: Script shell
chmod +x run.sh
./run.sh

# Option 2: Manuel
source .venv/bin/activate
streamlit run app.py
```

## Structure du Projet

```
tradingtool2/
├── app.py                 # Application Streamlit principale
├── requirements.txt       # Dépendances Python
├── README.md             # Documentation
├── launch.bat            # Script de lancement Windows
├── launch.ps1            # Script PowerShell
├── run.sh                # Script de lancement Linux/Mac
├── data/                 # Stockage des données Parquet
├── config/               # Fichiers de configuration
│   ├── settings.py       # Configuration centralisée
│   └── config.env        # Variables d'environnement
├── src/                  # Code source modulaire
│   ├── data/            # Gestion des données
│   ├── analysis/        # Analyses et calculs
│   ├── calendar/        # Calendriers de trading
│   └── utils/           # Utilitaires
└── tests/               # Tests unitaires
```

## Configuration

1. **Copier le fichier de configuration**
```bash
cp config.env .env
```

2. **Modifier les paramètres dans `.env`**
3. **Configurer les paramètres dans l'interface**

## Utilisation

1. **Dashboard** : Vue d'ensemble et métriques clés
2. **Gestion des Données** : Récupération et stockage des données
3. **Calendriers** : Configuration des heures de trading
4. **Analyses** : Calcul des moyennes et analyses
5. **Configuration** : Paramètres généraux et API

## Développement

- Architecture modulaire et extensible
- Fonctions réutilisables et bien documentées
- Tests unitaires pour chaque module
- Documentation technique complète

## Prochaines Étapes

- [x] Structure de base du projet
- [x] Interface Streamlit
- [x] Configuration centralisée
- [ ] Implémentation de la couche de données
- [ ] Gestion des calendriers de trading
- [ ] Calcul des moyennes automatiques
- [ ] Backtesting de stratégies
- [ ] Connexion IBKR
- [ ] Interface d'administration avancée

## Résolution des Problèmes

### ❌ Erreur "No module named 'distutils'"
```bash
python -m pip install setuptools wheel
```

### ❌ Erreur "streamlit is not recognized"
```bash
# Utiliser le chemin complet
.venv\Scripts\streamlit.exe run app.py
```

### ❌ Erreur de politique d'exécution PowerShell
```bash
# Utiliser le script batch ou contourner la politique
powershell -ExecutionPolicy Bypass -File launch.ps1
```

### ❌ Problèmes de dépendances
```bash
# Mettre à jour pip et installer setuptools
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Support

Pour toute question ou problème, consultez la documentation ou créez une issue sur le repository.
