# NEURAX - Akcelerator neuronskih mreža na DE1-SoC platformi

## Pregled

NEURAX je napredni akcelerator neuronskih mreža dizajniran za DE1-SoC platformu koja kombinuje ARM Cortex-A9 procesor (HPS) sa Cyclone V FPGA. Sistem omogućava efikasno izvršavanje inferecije neuronskih mreža kroz hardware akceleraciju ključnih operacija.

## Arhitektura

### Hardware komponente
- **FPGA Akcelerator**: Konfigurabilni akcelerator sa tri glavna bloka
  - 2D Konvolucijski blok
  - Aktivacioni blok (ReLU, Tanh, Sigmoid)
  - Pooling blok (Max/Average)
- **DMA kontroler**: Za efikasan prenos podataka
- **Registarski interfejs**: Avalon-MM/AXI-Lite za konfiguraciju

### Software komponente
- **NEURAX biblioteka**: C/C++ API za korišćenje akceleratora
- **Drajveri**: Linux kernel moduli za komunikaciju sa FPGA
- **Pomoćne biblioteke**: Predprocesiranje, postprocesiranje slike
- **Demo aplikacije**: Primeri korišćenja za različite NN zadatke

## Struktura projekta

```
neurax/
├── hardware/           # FPGA design fajlovi
│   ├── fpga/          # Verilog/VHDL implementacija
│   └── qsys/          # QSys sistem integracija
├── software/          # Software komponente
│   ├── lib/           # NEURAX biblioteka
│   ├── drivers/       # Kernel drajveri
│   └── utils/         # Pomoćne biblioteke
├── demo/              # Demo aplikacije
├── tests/             # Test suite
└── docs/              # Dokumentacija

```

## Ključne karakteristike

- **Konfigurabilnost**: Svi blokovi se mogu nezavisno konfigurisati
- **Fleksibilnost**: Podrška za 8-bit i 16-bit podatke
- **Standardni interfejsi**: Avalon-ST, AXI-Stream za podatke
- **Optimizacija**: Paralelizacija i pipeline obrada
- **Skalabilnost**: Modularni dizajn za lako proširivanje

## Specifikacije FPGA akceleratora

### Konvolucijski blok
- Konfigurabilne dimenzije kernela (do 11x11)
- Podrška za različite stride vrednosti
- Optimizovan za 2D konvoluciju

### Aktivacioni blok
- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic tangent)
- Sigmoid
- Runtime selekcija funkcije

### Pooling blok
- Max pooling
- Average pooling
- Konfigurabilne dimenzije prozora

## Instalacija i pokretanje

### Preduslov
- Intel Quartus Prime (za FPGA synthesis)
- Altera SoC EDS (za software development)
- Linux kernel headers
- OpenCV (za image processing)

### Build process
```bash
# Clone repository
git clone <repository-url>
cd neurax

# Build hardware
make hardware

# Build software
make software

# Run tests
make test
```

## Dokumentacija

Detaljnu dokumentaciju možete pronaći u `docs/` direktorijumu:
- Arhitektura sistema
- API referenca
- Tutorial za korišćenje
- Hardware specifikacije

## Licenca

[Dodati informacije o licenci]

## Kontakt

[Dodati kontakt informacije]
